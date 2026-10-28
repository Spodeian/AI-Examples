//! Archive and Compression Engine: TAR, TAR.GZ, and Zlib/Deflate container operations.

use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;

/// Compresses bytes using zlib/deflate compression.
pub fn compress_zlib(data: &[u8]) -> Vec<u8> {
    compress_to_vec_zlib(data, 6)
}

/// Decompresses bytes using zlib/deflate decompression.
pub fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    decompress_to_vec_zlib(data).map_err(|e| format!("Decompression failed: {:?}", e))
}

/// An entry within a TAR archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TarEntry {
    pub path: String,
    pub data: Vec<u8>,
}

/// Packages a list of files into standard USTAR TAR archive bytes.
pub fn create_tar(entries: &[TarEntry]) -> Vec<u8> {
    let mut tar = Vec::new();
    for entry in entries {
        let mut header = [0u8; 512];
        let path_bytes = entry.path.as_bytes();
        let name_len = path_bytes.len().min(99);
        header[0..name_len].copy_from_slice(&path_bytes[0..name_len]);

        header[100..108].copy_from_slice(b"0000644\0");
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");

        let size_octal = format!("{:011o}\0", entry.data.len());
        header[124..136].copy_from_slice(size_octal.as_bytes());

        header[136..148].copy_from_slice(b"00000000000\0");
        header[148..156].copy_from_slice(b"        ");
        header[156] = b'0'; // regular file

        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");

        let sum: u32 = header.iter().map(|&b| b as u32).sum();
        let chk_str = format!("{:06o}\0 ", sum);
        header[148..156].copy_from_slice(chk_str.as_bytes());

        tar.extend_from_slice(&header);
        tar.extend_from_slice(&entry.data);

        let pad = (512 - (entry.data.len() % 512)) % 512;
        if pad > 0 {
            tar.resize(tar.len() + pad, 0);
        }
    }
    tar.resize(tar.len() + 1024, 0);
    tar
}

/// Unpacks USTAR TAR archive bytes into individual file entries.
pub fn read_tar(bytes: &[u8]) -> Result<Vec<TarEntry>, String> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + 512 <= bytes.len() {
        let header = &bytes[offset..offset + 512];
        if header.iter().all(|&b| b == 0) {
            break; // End of archive marker
        }

        let name_end = header[0..100].iter().position(|&b| b == 0).unwrap_or(100);
        let path = String::from_utf8_lossy(&header[0..name_end])
            .trim()
            .to_string();

        let size_str = String::from_utf8_lossy(&header[124..136])
            .trim_matches(|c: char| c == '\0' || c == ' ')
            .to_string();
        let size = usize::from_str_radix(&size_str, 8)
            .map_err(|e| format!("Invalid octal size in TAR header: {}", e))?;

        offset += 512;
        if offset + size > bytes.len() {
            return Err("Unexpected EOF while reading TAR entry data".to_string());
        }

        let data = bytes[offset..offset + size].to_vec();
        entries.push(TarEntry { path, data });

        let pad = (512 - (size % 512)) % 512;
        offset += size + pad;
    }

    Ok(entries)
}

/// Creates a compressed TAR.GZ archive.
pub fn create_tar_gz(entries: &[TarEntry]) -> Vec<u8> {
    let tar = create_tar(entries);
    compress_zlib(&tar)
}

/// Reads a compressed TAR.GZ archive.
pub fn read_tar_gz(bytes: &[u8]) -> Result<Vec<TarEntry>, String> {
    let tar = decompress_zlib(bytes)?;
    read_tar(&tar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tar_roundtrip() {
        let entries = vec![
            TarEntry {
                path: "blueprint.aid.yaml".to_string(),
                data: b"model_id: test".to_vec(),
            },
            TarEntry {
                path: "manifest.json".to_string(),
                data: b"{\"version\": 1}".to_vec(),
            },
        ];

        let tar = create_tar(&entries);
        assert!(!tar.is_empty());

        let unpacked = read_tar(&tar).unwrap();
        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked[0].path, "blueprint.aid.yaml");
        assert_eq!(unpacked[0].data, b"model_id: test");
        assert_eq!(unpacked[1].path, "manifest.json");
    }

    #[test]
    fn test_tar_gz_roundtrip() {
        let entries = vec![TarEntry {
            path: "data.txt".to_string(),
            data: vec![42u8; 1000],
        }];

        let tar_gz = create_tar_gz(&entries);
        assert!(tar_gz.len() < 1000); // Compression verified

        let unpacked = read_tar_gz(&tar_gz).unwrap();
        assert_eq!(unpacked.len(), 1);
        assert_eq!(unpacked[0].data.len(), 1000);
    }
}
