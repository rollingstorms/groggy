// Test edge mapping with curvature
const testEdges = [
    { id: 0, source: 1, target: 2, attributes: {}, curvature: null },
    { id: 1, source: 1, target: 2, attributes: {}, curvature: 0.5 },
    { id: 2, source: 1, target: 2, attributes: {}, curvature: -0.5 },
    { id: 3, source: 2, target: 3, attributes: {}, curvature: null }
];

console.log('Testing edge mapping...');
console.log('Input edges:', JSON.stringify(testEdges, null, 2));

// Simulate the mapping from app.js
const mappedEdges = testEdges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    attributes: edge.attributes,
    color: edge.color,
    width: edge.width,
    opacity: edge.opacity,
    style: edge.style,
    curvature: edge.curvature
}));

console.log('\nMapped edges:', JSON.stringify(mappedEdges, null, 2));

// Check which have curvature
const withCurvature = mappedEdges.filter(e => e.curvature !== null && e.curvature !== undefined);
console.log(`\n✅ Edges with curvature: ${withCurvature.length}`);
withCurvature.forEach(e => {
    console.log(`   Edge ${e.id}: curvature = ${e.curvature}`);
});
