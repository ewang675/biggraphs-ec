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

    int prefix_sum[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); i++) {
        prefix_sum[i] = 0;
    }

    int sum = 0;
    for (int i = 0; i < omp_get_max_threads(); i++) {
        prefix_sum[i] = sum;
        sum += local_frontiers[i].count;
    }

    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) {
        memcpy(&frontier->vertices[prefix_sum[i]], local_frontiers[i].vertices, local_frontiers[i].count * sizeof(int));
    }

    frontier->count = sum;
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

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        #pragma omp parallel for
        for (int i = 0; i < omp_get_max_threads(); i++) {
            vertex_set_clear(&local_frontiers[i]);
        }

        top_down_step(graph, frontier, local_frontiers, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    }
}

void bottom_up_step(
    Graph g,
    bool* frontier,
    bool* new_frontier,
    bool* empty,
    int* distances)
{
    #pragma omp parallel for
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
                    if (__sync_bool_compare_and_swap(&distances[node], NOT_VISITED_MARKER, distances[incoming] + 1)) {
                        new_frontier[node] = true;
                        if (*empty) {
                            __sync_bool_compare_and_swap(empty, true, false);
                        }
                    }
                }
            }
        }

    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    bool * frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));
    bool * new_frontier = (bool *) (malloc(sizeof(bool) * graph->num_nodes));

    // initialize distances
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    sol->distances[ROOT_NODE_ID] = 0;

    // initialize frontier
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i ++) {
        frontier[i] = false;
    }
    frontier[ROOT_NODE_ID] = true; 

    bool empty = false;  
    
    while (!empty) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif 
        // clear variables
        empty = true;
        #pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i ++) {
            new_frontier[i] = false;
        }

        bottom_up_step(graph, frontier, new_frontier, &empty, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        
        bool* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

    }
    

}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
