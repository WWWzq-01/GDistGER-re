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
int train_corpus_cuda(int argc, char **argv,const vector<vertex_id_t>& degrees,SyncQueue& corpus_q,int _my_rank,myEdgeContainer *csr);

struct Empty
{
};

// ./bin/simple_walk -g ./karate.data -v 34 -w 34 -o ./out/walks.txt > perf_dist.txt
int main(int argc, char **argv)
{
    umask(0);
    Timer timer;
    double load_graph_time = 0.0;
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
    printf("[ %d ] myec copy ok\n",my_rank);
    // cout <<my_rank <<" myec access " << myec-> adj_lists[110].begin->neighbour<<endl; 
    // cout << my_rank <<" graph.csr access " << graph.csr-> adj_lists[110].begin->neighbour<<endl; 

    // train_corpus_cuda(argc,argv,vertex_degree,graph.out_queue,my_rank,myec);
    thread train_thread(train_corpus_cuda,argc,argv,std::ref(vertex_degree),std::ref(graph.out_queue), my_rank,myec);
    // * 

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

    WalkerConfig<real_t, uint32_t> walker_conf(opt.walker_num);
    TransitionConfig<real_t, uint32_t> tr_conf(extension_comp);
    for (int i = 0; i < 1; i++) // ???????????? for(int i = 0; i < 1; i++) 
    {
        int pid = get_mpi_rank();
        WalkConfig walk_conf;
        if (!opt.output_path.empty())
        {
            std::cout<< opt.output_path <<std::endl;
            walk_conf.set_output_file(opt.output_path.c_str());
        }
        if (opt.set_rate)
        {
            walk_conf.set_walk_rate(opt.rate);
        }
        Timer walk_timer;
        printf("=================[ %d ] RANDOM WALK ================\n",my_rank);
        graph.random_walk(&walker_conf, &tr_conf, &walk_conf);
        double sum_time = walk_timer.duration();
        double walk_time = sum_time - graph.other_time;
        printf("[p%u][sum time:]%lf [walk time:]%lf [other time:]%lf\n", graph.get_local_partition_id(), sum_time, walk_time, graph.other_time);
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


    MPI_Allreduce(MPI_IN_PLACE,graph.vertex_cn.data(), graph.get_vertex_num(), get_mpi_data_type<int>(), MPI_SUM, MPI_COMM_WORLD);

    // ================= annotation line ====================

   // Test for bitmap
    compress_t compress_corpus;
    CorpusCompressor compressor;
    compressor.compressCorpus(graph.local_corpus, compress_corpus);

    size_t origin_size = 0;
    for(size_t i = 0; i < graph.local_corpus.size();i++){
        origin_size += graph.local_corpus[i].size();
    }
    origin_size *= sizeof(vertex_id_t);

    // cout << "original size: " << origin_size << " Byte." << endl;
    size_t compress_size = 0;
    for(size_t i = 0; i < compress_corpus.size();i++){
        compress_size += compress_corpus[i].coreMap.mem_size();
        compress_size += compress_corpus[i].misc_data.size() * sizeof(vertex_id_t);
    }
    cout << "Original size: " << origin_size * 4 << " Byte." << endl;
    cout <<"Top compress size: " << compress_size << " Byte." << endl;
    cout <<"Top Ratio: " << (float)compress_size/origin_size << endl;

    origin_size = 0;
    compress_size =0;
    for(size_t i = 0; i < graph.local_corpus.size();i++) {
        origin_size += graph.local_corpus[i].size();
        compress_size += graph.local_corpus[i].size();
        map<vertex_id_t,int> freq;
        for(size_t j = 1; j < graph.local_corpus[i].size();j++){
            freq[graph.local_corpus[i][j]]++;
        }
        int max_freq = 0;
        vector<pair<vertex_id_t,int>> core_array;
        for(auto& pair: freq){
            core_array.push_back(pair);
        }
        sort(core_array.begin(),core_array.end(),[](pair<vertex_id_t, int>&p1,pair<vertex_id_t,int>&p2){
            return p1.second > p2.second;
        });
        if(core_array.size()>0) compress_size -= core_array[0].second;
        if(core_array.size()>1) compress_size -= core_array[1].second;
    }
    cout << "Original size: " << origin_size * 4 << " Byte." << endl;
    cout <<"Theory compress size: " << compress_size * 4 << " Byte." << endl;
    cout <<"Ratio: " << (float)compress_size/origin_size << endl;

    train_thread.join();
    printf("> [p%d WHOLE TIME:] %lf \n",get_mpi_rank(), timer.duration());
    printf("msgTime： %lf \n",graph.msg_time);
    printf("load graph time： %lf \n",load_graph_time);
    // train_corpus_cuda(argc,argv,vertex_degree,graph.out_queue);
    // dsgl(argc, argv,&graph.vertex_cn,&graph.new_sort,&graph);
    return 0;
}
