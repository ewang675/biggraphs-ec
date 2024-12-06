#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define CACHE_LINE_SIZE 64          // from googling the architecture
#define NUM_LINES_PER_CHUNK 32     // from experimenting
#define THRESHOLD 0.25

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* local_frontiers,
    int* distances)
{

    #pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER && __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
                local_frontiers[omp_get_thread_num()].vertices[local_frontiers[omp_get_thread_num()].count] = outgoing;
                local_frontiers[omp_get_thread_num()].count++; 
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set* frontier = &list1;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // setup local frontiers for each thread
    vertex_set local_frontiers[omp_get_max_threads()];
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        vertex_set_init(&local_frontiers[i], graph->num_nodes);
    }
    
    while (frontier->count != 0) {

        /*
        double start_time = CycleTimer::currentSeconds();
        */

        #pragma omp parallel for
        for (int i = 0; i < omp_get_max_threads(); i++) {
            vertex_set_clear(&local_frontiers[i]);
        }

        top_down_step(graph, frontier, local_frontiers, sol->distances);

        /*
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
        printf("frontier has %d nodes, representing %f of total nodes\n", frontier->count, (float)frontier->count / graph->num_nodes);
        */

        // accumulate prefix sum of count of nodes in each frontier
        int prefix_sum[omp_get_max_threads()];
        for (int i = 0; i < omp_get_max_threads(); i++) {
            prefix_sum[i] = 0;
        }
        int sum = 0;
        for (int i = 0; i < omp_get_max_threads(); i++) {
            prefix_sum[i] = sum;
            sum += local_frontiers[i].count;
        }

        // copy each local_frontier into frontier at the appropriate index
        #pragma omp parallel for
        for (int i = 0; i < omp_get_max_threads(); i++) {
            memcpy(&frontier->vertices[prefix_sum[i]], local_frontiers[i].vertices, local_frontiers[i].count * sizeof(int));
        }

        frontier->count = sum;
    }
}

bool bottom_up_step(
    Graph g,
    bool* frontier,
    bool* new_frontier,
    int* distances)
{
    bool cont = 0; 
    // chunk_size ensures sequential accesses in frontier and new_frontier
    #pragma omp parallel for schedule(dynamic, CACHE_LINE_SIZE * NUM_LINES_PER_CHUNK)
    for (int node = 0; node < g->num_nodes; node++) {
        if (distances[node] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];

            // check if there is an incoming edge from a frontier node
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                if (frontier[incoming]) {
                    distances[node] = distances[incoming] + 1; 
                    new_frontier[node] = true;
                    if (!cont) { 
                        #pragma omp critical
                        { 
                            cont = true; 
                        }
                    }
                    break;
                }
            }
        }
    }
    return cont;
}

void bfs_bottom_up(Graph graph, solution* sol) {
    bool * frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));
    bool * new_frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));

    // initialize distances
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    sol->distances[ROOT_NODE_ID] = 0;

    // initialize frontier
    #pragma omp parallel for schedule(static, CACHE_LINE_SIZE)
    for (int i = 0; i < graph->num_nodes; i ++) {
        frontier[i] = false;
    }
    frontier[ROOT_NODE_ID] = true;
     
    bool cont = true;  
    while (cont) {

        /*
        double start_time = CycleTimer::currentSeconds();
        */

        // clear variables
        #pragma omp parallel for schedule(static, CACHE_LINE_SIZE)
        for (int i = 0; i < graph->num_nodes; i ++) {
            new_frontier[i] = false;
        }
        cont = bottom_up_step(graph, frontier, new_frontier, sol->distances);

        /*
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier_size, end_time - start_time);
        printf("frontier has %d nodes, representing %f of total nodes\n", frontier_size, (float)frontier_size / graph->num_nodes);
        */

        bool* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(frontier); 
    free(new_frontier);
}

void bfs_hybrid(Graph graph, solution* sol) {
    // This solution switches from TD to BU, but does not switch back.
    // The bet is that the advantage that TD has over BU in the final few steps
    // of BFS is not worth the overhead of transferring data back to the
    // TD data structures.

    bool just_switched = true;  // indicates if we just switched from TD to BU
    bool use_TD = true;         // indicates if we should use TD on this step

    // ---------- TOP-DOWN INIT ---------- // 
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set* TD_frontier = &list1;

    // setup frontier with the root node
    TD_frontier->vertices[TD_frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // setup local frontiers for each thread
    vertex_set TD_local_frontiers[omp_get_max_threads()];
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        vertex_set_init(&TD_local_frontiers[i], graph->num_nodes);
    }
    
    // ---------- BOTTOM-UP INIT ---------- // 
    bool * BU_frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));
    bool * BU_new_frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));

    // initialize frontier
    #pragma omp parallel for schedule(static, CACHE_LINE_SIZE)
    for (int i = 0; i < graph->num_nodes; i ++) {
        BU_frontier[i] = false;
    }
    BU_frontier[ROOT_NODE_ID] = true;

    bool cont = true; 

    // ---------- SOLUTION INIT ---------- // 
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    sol->distances[ROOT_NODE_ID] = 0;

    // ---------- PERFORM BFS ---------- // 
    while (cont) {
        /*
        double start_time = CycleTimer::currentSeconds();
        */
        if (use_TD) {
            #pragma omp parallel for
            for (int i = 0; i < omp_get_max_threads(); i++) {
                vertex_set_clear(&TD_local_frontiers[i]);
            }

            top_down_step(graph, TD_frontier, TD_local_frontiers, sol->distances);
            cont = TD_frontier->count;

            // accumulate prefix sum of count of nodes in each frontier
            int prefix_sum[omp_get_max_threads()];
            for (int i = 0; i < omp_get_max_threads(); i++) {
                prefix_sum[i] = 0;
            }
            int sum = 0;
            for (int i = 0; i < omp_get_max_threads(); i++) {
                prefix_sum[i] = sum;
                sum += TD_local_frontiers[i].count;
            }

            // copy each local_frontier into frontier at the appropriate index
            #pragma omp parallel for
            for (int i = 0; i < omp_get_max_threads(); i++) {
                memcpy(&TD_frontier->vertices[prefix_sum[i]], TD_local_frontiers[i].vertices, TD_local_frontiers[i].count * sizeof(int));
            }

            TD_frontier->count = sum;

            if (float(TD_frontier->count) / graph->num_nodes > THRESHOLD) { use_TD = false; }
        } else {
            if (just_switched) {
                #pragma omp parallel for
                for (int i = 0; i < TD_frontier->count; i ++) {
                    BU_frontier[TD_frontier->vertices[i]] = true;
                }
                just_switched = false; 
            }
            #pragma omp parallel for schedule(static, CACHE_LINE_SIZE)
            for (int i = 0; i < graph->num_nodes; i ++) {
                BU_new_frontier[i] = false;
            }

            cont = bottom_up_step(graph, BU_frontier, BU_new_frontier, sol->distances);

            bool* tmp = BU_frontier;
            BU_frontier = BU_new_frontier;
            BU_new_frontier = tmp;
        }
        /*
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier_size, end_time - start_time);
        printf("frontier has %d nodes, representing %f of total nodes\n", frontier_size, (float)frontier_size / graph->num_nodes);
        */ 
    }
    free(BU_frontier); 
    free(BU_new_frontier);
}
