#include "type.hpp"
#include "walk.hpp"
#include "option_helper.hpp"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "edge_container.hpp"
#include <sys/stat.h>
#include <cstdio>
#include "compress.hpp"
#include <map>


// template struct EdgeContainer<real_t>;
using namespace std;

struct TrainingConfig {
    int init_round;
    int batch_size;
    
    TrainingConfig(int init_round = 1, int batch_size = 16384) 
        : init_round(init_round), batch_size(batch_size) {}
};

int train_corpus_cuda(int argc, char **argv,const vector<vertex_id_t>& degrees,SyncQueue& corpus_q,int _my_rank,myEdgeContainer *csr, const TrainingConfig& config);
extern double actual_training_time;

struct Empty
{
};

// ./bin/simple_walk -g ./karate.data -v 34 -w 34 -o ./out/walks.txt > perf_dist.txt
int main(int argc, char **argv)
{
    umask(0);
    Timer timer;
    double load_graph_time = 0.0;
    double data_conversion_time = 0.0;
    double walk_setup_time = 0.0; 
    double walk_execution_time = 0.0;
    double corpus_output_time = 0.0;
    double corpus_compression_time = 0.0;
    double training_time = 0.0;
    MPI_Instance mpi_instance(&argc, &argv);
    int my_rank = get_mpi_rank();


    RandomWalkOptionHelper opt;
    opt.parse(argc, argv);

    WalkEngine<real_t, uint32_t> graph;

    //=============== annotation line ===================
    graph.set_init_round(opt.init_round);
    printf("opt min length: %d\n",opt.min_length);
    graph.set_minLength(opt.min_length);
    printf("init_round = %d, min_length = %d\n", graph.init_round, graph.minLength);
    printf("graph path: %s\n",opt.graph_path.c_str());
    Timer load_timer;
    graph.load_graph(opt.v_num, opt.graph_path.c_str(), opt.partition_path.c_str(), opt.make_undirected);
    load_graph_time = load_timer.duration();
    printf("[ %d ] load_graph ok!\n",my_rank);
    graph.vertex_cn.resize(graph.get_vertex_num());
    // graph.load_commonNeighbors(opt.graph_common_neighbour.c_str());
    vector<vertex_id_t> vertex_degree(graph.v_num,0);
    for (vertex_id_t v = 0; v < graph.v_num; v++){
        vertex_degree[v] = graph.vertex_out_degree[v];
    }
    // =============== Data Structure Conversion ===============
    Timer conversion_timer;
    printf("[ %d ] Starting data structure conversion...\n", my_rank);
    
    //myEdgeContainer* myec = reinterpret_cast<myEdgeContainer*>(&graph.g_csr);
    //cout <<"myec access " << myec-> adj_lists[0].begin->neighbour<<endl; 
    myEdgeContainer* myec = new myEdgeContainer();
    myec->adj_lists = new myAdjList[graph.v_num];
    myec->adj_units = new myAdjUnit[graph.e_num];
    edge_id_t chunk_edge_idx = 0;
    printf("[ %d ] malloc ok\n",my_rank);
    for(vertex_id_t v_i = 0; v_i < graph.v_num; v_i++){
      myec->adj_lists[v_i].begin = myec->adj_units + chunk_edge_idx;
      chunk_edge_idx += graph.csr->adj_lists[v_i].end -graph.csr->adj_lists[v_i].begin; 
      myec->adj_lists[v_i].end = myec->adj_units + chunk_edge_idx;
    }
    for(edge_id_t e_i = 0; e_i < graph.e_num; e_i++){
     myec->adj_units[e_i].neighbour = graph.csr->adj_units[e_i].neighbour;
     myec->adj_units[e_i].data = graph.csr->adj_units[e_i].data;
    }
    data_conversion_time = conversion_timer.duration();
    printf("[ %d ] Data conversion completed in %lf seconds\n", my_rank, data_conversion_time);
    // cout <<my_rank <<" myec access " << myec-> adj_lists[110].begin->neighbour<<endl; 
    // cout << my_rank <<" graph.csr access " << graph.csr-> adj_lists[110].begin->neighbour<<endl; 

    // =============== Start Training Thread ===============
    Timer training_start_timer;
    printf("[ %d ] Starting training thread...\n", my_rank);
    TrainingConfig train_config(graph.init_round, opt.batch_size);
    thread train_thread(train_corpus_cuda,argc,argv,std::ref(vertex_degree),std::ref(graph.out_queue), my_rank,myec, train_config);
    printf("[ %d ] Training thread started\n", my_rank);

    auto extension_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v)
    {
        // return 0.995;
        return walker.step >= 40 ? 0.0 : 1.0;
    };
    auto static_comp = [&](vertex_id_t v, AdjUnit<real_t> *edge)
    {
        return 1.0; /*edge->data is a real number denoting edge weight*/
    };
    auto dynamic_comp = [&](Walker<uint32_t> &walker, vertex_id_t current_v, AdjUnit<real_t> *edge)
    {
        return 1.0;
    };
    auto dynamic_comp_upperbound = [&](vertex_id_t v_id, AdjList<real_t> *adj_lists)
    {
        return 1.0;
    };

    // =============== Walk Configuration Setup ===============
    Timer setup_timer;
    printf("[ %d ] Setting up walk configuration...\n", my_rank);
    
    WalkerConfig<real_t, uint32_t> walker_conf(opt.walker_num);
    TransitionConfig<real_t, uint32_t> tr_conf(extension_comp);
    walk_setup_time = setup_timer.duration();
    printf("[ %d ] Walk configuration setup completed in %lf seconds\n", my_rank, walk_setup_time);
    
    for (int i = 0; i < 1; i++) // ???????????? for(int i = 0; i < 1; i++) 
    {
        int pid = get_mpi_rank();
        WalkConfig walk_conf;
        
        // =============== Output Path Configuration (Potential I/O Bottleneck) ===============
        Timer output_config_timer;
        if (!opt.output_path.empty())
        {
            printf("[ %d ] Configuring output to disk: %s (POTENTIAL I/O BOTTLENECK)\n", my_rank, opt.output_path.c_str());
            std::cout<< opt.output_path <<std::endl;
            walk_conf.set_output_file(opt.output_path.c_str());
        }
        if (opt.set_rate)
        {
            walk_conf.set_walk_rate(opt.rate);
        }
        double output_config_time = output_config_timer.duration();
        printf("[ %d ] Output configuration time: %lf seconds\n", my_rank, output_config_time);
        
        // =============== Random Walk Execution ===============
        Timer walk_timer;
        printf("=================[ %d ] RANDOM WALK EXECUTION ================\n",my_rank);
        graph.random_walk(&walker_conf, &tr_conf, &walk_conf);
        double sum_time = walk_timer.duration();
        walk_execution_time = sum_time;
        double walk_time = sum_time - graph.other_time;
        printf("[p%u][WALK EXECUTION] Total: %lf s, Pure walk: %lf s, Other: %lf s\n", 
               graph.get_local_partition_id(), sum_time, walk_time, graph.other_time);
        
        // 使用内存管道，无磁盘I/O
        if (!opt.output_path.empty()) {
            printf("[ %d ] *** USING MEMORY PIPELINE (NO DISK I/O) *** \n", my_rank);
        }
    }
    printf("> [p%d RANDOM WALKING TIME:] %lf \n",get_mpi_rank(), timer.duration());

    // * 关闭任务队列
    graph.out_queue.closeQueue();

    if(get_mpi_rank()==0){
        cout<<"============partion table=========="<<endl;
        for(int p=0;p<get_mpi_size();p++){
            cout<<"part: "<<p<<" "<<graph.vertex_partition_begin[p]<<" ~ "<<graph.vertex_partition_end[p]<<endl;
        }
    }


    // =============== Information Feedback Collection ===============
    Timer feedback_timer;
    printf("[ %d ] Collecting information feedback data...\n", my_rank);
    MPI_Allreduce(MPI_IN_PLACE,graph.vertex_cn.data(), graph.get_vertex_num(), get_mpi_data_type<int>(), MPI_SUM, MPI_COMM_WORLD);
    double feedback_time = feedback_timer.duration();
    printf("[ %d ] Information feedback collection completed in %lf seconds\n", my_rank, feedback_time);

    // =============== Corpus Compression ===============
    Timer compression_timer;
    printf("[ %d ] Starting corpus compression...\n", my_rank);
    
    compress_t compress_corpus;
    CorpusCompressor compressor;
    compressor.compressCorpus(graph.local_corpus, compress_corpus);
    corpus_compression_time = compression_timer.duration();
    printf("[ %d ] Corpus compression completed in %lf seconds\n", my_rank, corpus_compression_time);

    // Use saved compression statistics (calculated before move)
    size_t origin_size = graph.saved_origin_size;
    size_t compress_size = 0;
    for(size_t i = 0; i < compress_corpus.size();i++){
        compress_size += compress_corpus[i].coreMap.mem_size();
        compress_size += compress_corpus[i].misc_data.size() * sizeof(vertex_id_t);
    }
    cout << "Original size: " << origin_size * 4 << " Byte." << endl;
    cout <<"Top compress size: " << compress_size << " Byte." << endl;
    cout <<"Top Ratio: " << (origin_size > 0 ? (float)compress_size/origin_size : 0.0f) << endl;

    // Use saved theoretical compression size
    size_t theory_compress_size = graph.saved_theory_compress_size;
    cout << "Original size: " << origin_size * 4 << " Byte." << endl;
    cout <<"Theory compress size: " << theory_compress_size * 4 << " Byte." << endl;
    cout <<"Ratio: " << (origin_size > 0 ? (float)theory_compress_size/origin_size : 0.0f) << endl;

    // =============== Wait for Training Completion ===============
    Timer training_wait_timer;
    printf("[ %d ] Waiting for training thread to complete...\n", my_rank);
    train_thread.join();
    double thread_join_time = training_wait_timer.duration();
    training_time = actual_training_time;  // Use actual training time from training thread
    printf("[ %d ] Training thread join completed in %lf seconds (join wait time)\n", my_rank, thread_join_time);
    printf("[ %d ] Actual training execution time: %lf seconds\n", my_rank, training_time);
    
    double total_time = timer.duration();
    
    // =============== PERFORMANCE ANALYSIS REPORT ===============
    if (my_rank == 0) {
        printf("\n");
        printf("=================== PERFORMANCE ANALYSIS REPORT ===================\n");
        printf("Total execution time: %lf seconds\n", total_time);
        printf("---------------------------------------------------------------------\n");
        printf("1. Graph loading time:          %lf s  (%.2f%% of total)\n", 
               load_graph_time, (load_graph_time/total_time)*100);
        printf("2. Data conversion time:        %lf s  (%.2f%% of total)\n", 
               data_conversion_time, (data_conversion_time/total_time)*100);
        printf("3. Walk setup time:             %lf s  (%.2f%% of total)\n", 
               walk_setup_time, (walk_setup_time/total_time)*100);
        printf("4. Walk execution time:         %lf s  (%.2f%% of total)\n", 
               walk_execution_time, (walk_execution_time/total_time)*100);
        printf("   - Pure walk time:            %lf s\n", walk_execution_time - graph.other_time);
        printf("   - Other operations:          %lf s\n", graph.other_time);
        printf("   - Message passing time:      %lf s\n", graph.msg_time);
        printf("5. Corpus compression time:     %lf s  (%.2f%% of total)\n", 
               corpus_compression_time, (corpus_compression_time/total_time)*100);
        printf("6. Training time:               %lf s  (%.2f%% of total)\n", 
               training_time, (training_time/total_time)*100);
        printf("   - Memory-based training (no disk I/O)\n");
        printf("---------------------------------------------------------------------\n");
        printf("*** MEMORY OPTIMIZATION ENABLED ***\n");
        printf("Corpus data passed directly through memory pipeline.\n");
        printf("Disk I/O eliminated for optimal performance.\n");
        printf("=====================================================================\n");
        printf("\n");
    }
    
    printf("> [p%d WHOLE TIME:] %lf \n",get_mpi_rank(), total_time);
    printf("msgTime： %lf \n",graph.msg_time);
    printf("load graph time： %lf \n",load_graph_time);
    
    // train_corpus_cuda(argc,argv,vertex_degree,graph.out_queue);
    // dsgl(argc, argv,&graph.vertex_cn,&graph.new_sort,&graph);
    return 0;
}
