#include <metal_stdlib>
using namespace metal;

struct Edge {
    uint src;
    uint dst;
    float weight;
};

kernel void bellmanFord(
    device const Edge* edges [[ buffer(0) ]],
    device const uint* nodeEdgeStart [[ buffer(1) ]],
    device const float* distances [[ buffer(2) ]],
    device float* newDistances [[ buffer(3) ]],
    constant uint& numNodes [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= numNodes) return;
    //starts with current distance
    float minDistance = distances[tid];
    uint start = nodeEdgeStart[tid];
    uint end = nodeEdgeStart[tid + 1];

    for (uint i = start; i < end; i++) {
        Edge edge = edges[i];
        float dist_u = distances[edge.src];
        float newDist = dist_u + edge.weight;
        if (newDist < minDistance) {
            minDistance = newDist;
        }
    }
    newDistances[tid] = minDistance;
}
