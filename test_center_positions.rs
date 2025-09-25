// Simple test file to verify center_and_fit_positions works
use groggy::viz::streaming::data_source::Position;

fn center_and_fit_positions(
    mut positions: Vec<Position>,
    target_radius_px: Option<f64>,
) -> Vec<Position> {
    if positions.is_empty() { return positions; }

    // bbox
    let (mut xmin, mut xmax, mut ymin, mut ymax) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in &positions {
        if p.x.is_finite() { xmin = xmin.min(p.x); xmax = xmax.max(p.x); }
        if p.y.is_finite() { ymin = ymin.min(p.y); ymax = ymax.max(p.y); }
    }
    let cx = (xmin + xmax) * 0.5;
    let cy = (ymin + ymax) * 0.5;

    // translate to origin
    for p in &mut positions {
        p.x -= cx;
        p.y -= cy;
    }

    if let Some(target) = target_radius_px {
        // half-diagonal of bbox (farthest corner from origin)
        let hw = (xmax - xmin) * 0.5;
        let hh = (ymax - ymin) * 0.5;
        let half_diag = (hw*hw + hh*hh).sqrt().max(1e-9);
        let s = target / half_diag;
        for p in &mut positions {
            p.x *= s;
            p.y *= s;
        }
    }

    positions
}

fn main() {
    let positions = vec![
        Position { x: 100.0, y: 50.0 },
        Position { x: 200.0, y: 150.0 },
        Position { x: 300.0, y: -50.0 },
    ];
    let target = 250.0;
    let out = center_and_fit_positions(positions, Some(target));

    // Test centering (centroid ~ (0,0))
    let (mut sx, mut sy) = (0.0, 0.0);
    for p in &out { sx += p.x; sy += p.y; }
    let n = out.len() as f64;
    let (mx, my) = (sx / n, sy / n);
    
    println!("Centroid: ({:.6}, {:.6})", mx, my);
    assert!(mx.abs() < 1e-6, "mx={mx}");
    assert!(my.abs() < 1e-6, "my={my}");

    // Test scaling (half-diagonal ≤ target + epsilon)
    let (mut xmin, mut xmax, mut ymin, mut ymax) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in &out {
        if p.x.is_finite() { xmin = xmin.min(p.x); xmax = xmax.max(p.x); }
        if p.y.is_finite() { ymin = ymin.min(p.y); ymax = ymax.max(p.y); }
    }
    let hw = (xmax - xmin) * 0.5;
    let hh = (ymax - ymin) * 0.5;
    let half_diag = (hw*hw + hh*hh).sqrt();
    
    println!("Half diagonal: {:.3}, Target: {:.3}", half_diag, target);
    assert!(half_diag <= target * 1.0001, "half_diag={half_diag} > target={target}");
    
    println!("✅ All tests passed! center_and_fit_positions works correctly.");
    
    // Print final positions
    for (i, p) in out.iter().enumerate() {
        println!("Position {}: ({:.2}, {:.2})", i, p.x, p.y);
    }
}
