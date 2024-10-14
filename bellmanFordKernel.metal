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
    device atomic_float* newDistances [[ buffer(3) ]],
    constant uint& numNodes [[ buffer(4) ]],
    device atomic_uint* updatedFlag [[ buffer(5) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= numNodes) return;

    float dist_u = distances[tid];

    // If the distance to node 'u' is INFINITY, skip
    if (dist_u == INFINITY) return;

    // Get the start and end indices for the current node's edges
    uint start = nodeEdgeStart[tid];
    uint end = nodeEdgeStart[tid + 1];

    // Iterate over all outgoing edges for node 'u'
    for (uint i = start; i < end; i++) {
        Edge edge = edges[i];
        uint v = edge.dst;
        float newDist = dist_u + edge.weight;

        float oldDist = atomic_load_explicit(&newDistances[v], memory_order_relaxed);

        while (newDist < oldDist) {
            bool exchanged = atomic_compare_exchange_weak_explicit(
                &newDistances[v],
                &oldDist,
                newDist,
                memory_order_relaxed,
                memory_order_relaxed
            );
            if (exchanged) {
                // Setting the atomic flag to 1 shows an update occurred
                atomic_store_explicit(updatedFlag, 1, memory_order_relaxed);
                break;
            }
            // If not exchanged, oldDist has been updated, so we need to check the condition again
            // atomic_compare_exchange_weak_explicit updates oldDist on failure
            if (!(newDist < oldDist)) {
                break;
            }
        }
    }
}

