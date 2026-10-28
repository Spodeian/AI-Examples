use criterion::{Criterion, black_box, criterion_group, criterion_main};
use shared::export::{export_to_compressed_bson, export_to_csv, export_to_json};
use shared::models::ItemCollection;

fn bench_serialization(c: &mut Criterion) {
    let mut collection = ItemCollection::default();
    for i in 0..100 {
        collection.add(
            format!("Item Title {}", i),
            format!("Detailed description for item number {}", i),
            shared::Priority::Medium,
        );
    }

    c.bench_function("export_to_json_100_items", |b| {
        b.iter(|| {
            let res = export_to_json(black_box(&collection));
            let _ = black_box(res);
        })
    });

    c.bench_function("export_to_csv_100_items", |b| {
        b.iter(|| {
            let res = export_to_csv(black_box(&collection));
            black_box(res);
        })
    });

    c.bench_function("export_to_compressed_bson_100_items", |b| {
        b.iter(|| {
            let res = export_to_compressed_bson(black_box(&collection));
            let _ = black_box(res);
        })
    });
}

criterion_group!(benches, bench_serialization);
criterion_main!(benches);
