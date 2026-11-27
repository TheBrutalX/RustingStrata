mod common;
use rusting_strata::gguf::Gguf;
use rusting_strata::gpu_model::GpuModel;
use rusting_strata::model::Model;
use rusting_strata::slot_archive::{start_async_segmented_save, Archive, ArchiveConfig};
use rusting_strata::slot_segments::SegmentStore;
use rusting_strata::tokenizer::Tokenizer;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Manual phase breakdown for the archive save. Run with:
/// `cargo test --release --test archive_phase_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn archive_phase_breakdown() {
    let _gpu = common::gpu_lock();
    let Some(path) = common::model_path() else { return };
    let g = Gguf::open(&path).unwrap();
    let m = Model::load(&g).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let para = "A shared inference server keeps several conversations in GPU memory at once. Each slot owns its \
         attention keys and values, the recurrent state of its layers, and the token history. ";
    let mut text = String::new();
    while text.len() < 120_000 {
        text.push_str(para);
    }
    let ids = tok.encode(&text).unwrap();
    let ids = &ids[..ids.len().min(24_000)];
    eprintln!("prompt tokens: {}", ids.len());

    let Some((name, gpu)) = common::backends().into_iter().next() else { return };
    eprintln!("backend: {name}");
    let mut gm = GpuModel::load(&m, gpu, 32768, 0, 512 << 20, 0).unwrap();
    let t = Instant::now();
    gm.prefill(ids).unwrap();
    eprintln!("prefill: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let fingerprint = gm.slot_fingerprint();

    let t = Instant::now();
    let mut full = Vec::new();
    gm.save_slot_to(0, &mut full).unwrap();
    eprintln!("full stream GPU->host: {:.1} ms, {} bytes", t.elapsed().as_secs_f64() * 1e3, full.len());

    let t = Instant::now();
    let mut payloads = Vec::new();
    for start in (0..ids.len()).step_by(1024) {
        let end = (start + 1024).min(ids.len());
        let mut payload = Vec::new();
        gm.save_kv_segment(0, start, end, &mut payload).unwrap();
        payloads.push((start, end, payload));
    }
    let seg_bytes: usize = payloads.iter().map(|(_, _, p)| p.len()).sum();
    eprintln!("segments GPU->host: {:.1} ms, {} segments, {} bytes",
        t.elapsed().as_secs_f64() * 1e3, payloads.len(), seg_bytes);

    let t = Instant::now();
    let mut state = Vec::new();
    gm.save_recurrent_to(0, &mut state).unwrap();
    eprintln!("recurrent GPU->host: {:.1} ms, {} bytes", t.elapsed().as_secs_f64() * 1e3, state.len());

    let root = std::env::temp_dir().join(format!("strata-phase-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let t = Instant::now();
    let mut store = SegmentStore::open(root.join("segments")).unwrap();
    let mut parent: Option<String> = None;
    for (start, end, payload) in &payloads {
        let seg = store.put(fingerprint, parent.as_deref(), &ids[*start..*end], payload).unwrap();
        parent = Some(seg.id);
    }
    eprintln!("segment store disk: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    let state_path = root.join("state.bin");
    let mut f = std::fs::File::create(&state_path).unwrap();
    f.write_all(&state).unwrap();
    f.sync_all().unwrap();
    eprintln!("state file write+sync: {:.1} ms, {} bytes", t.elapsed().as_secs_f64() * 1e3, state.len());

    let t = Instant::now();
    let mut archive = Archive::open(ArchiveConfig {
        dir: root.join("archive"), budget_bytes: 8 << 30, min_tokens: 64, min_delta: 1, versions: 1,
    }).unwrap();
    let entry = archive.save_segmented(&mut gm, 0, 0, "phase-bench").unwrap();
    eprintln!("archive save_segmented total: {:.1} ms, entry {} bytes, {} segments",
        t.elapsed().as_secs_f64() * 1e3, entry.bytes, entry.segments.len());

    let async_archive = Arc::new(Mutex::new(Archive::open(ArchiveConfig {
        dir: root.join("archive-async"), budget_bytes: 8 << 30, min_tokens: 64, min_delta: 1, versions: 1,
    }).unwrap()));
    let t = Instant::now();
    let (mut sink, handle) = start_async_segmented_save(async_archive.clone(), 0, "async-phase").unwrap();
    gm.save_segmented_to(0, 0, &mut sink, &mut |_| Ok(())).unwrap();
    let gpu_and_send = t.elapsed();
    let async_entry = handle.join().unwrap();
    eprintln!("archive async save: gpu+send {:.1} ms, total {:.1} ms, entry {} bytes",
        gpu_and_send.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3, async_entry.bytes);
    eprintln!("summary: full={} segments={} state={} entry={}", full.len(), seg_bytes, state.len(), entry.bytes);

    drop(gm);
    let _ = std::fs::remove_dir_all(&root);
}
