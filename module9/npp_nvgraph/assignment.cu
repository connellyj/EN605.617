#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <chrono>
#include <string.h>
#include <fstream>
#include <iostream>

#include <npp.h>
#include <nvgraph.h>

void donpp()
{
    try
    {
        // load image from disk to host then device
        npp::ImageCPU_8u_C1 oHostSrc;
        npp::loadImage("lena_gray.bmp", oHostSrc);
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create necessary structs
        NppiSize oMaskSize = { 5, 5 };
        NppiSize oSrcSize = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
        NppiPoint oSrcOffset = { 0, 0 };
        NppiSize oSizeROI = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
        NppiPoint oAnchor = { oMaskSize.width / 2, oMaskSize.height / 2 };

        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

        // run box filter
        nppiFilterBoxBorder_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset, oDeviceDst.data(),
            oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE);

        // dcopy result back to host
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        const char* outfile = "out.pgm";
        npp::saveImage(outfile, oHostDst);
        std::cout << "Saved image: " << outfile << std::endl;
    }
    catch (npp::Exception& rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
}

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

void donvgraph()
{
    const size_t  n = 6, nnz = 10, vertex_numsets = 3, edge_numsets = 1;
    const float alpha1 = 0.85, alpha2 = 0.90;
    const void* alpha1_p = (const void*)&alpha1, * alpha2_p = (const void*)&alpha2;

    // Allocate host data
    cudaDataType_t edge_dimT = CUDA_R_32F;
    int* destination_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    int* source_indices_h = (int*)malloc(nnz * sizeof(int));
    float* weights_h = (float*)malloc(nnz * sizeof(float));
    float* bookmark_h = (float*)malloc(n * sizeof(float));
    float* pr_1 = (float*)malloc(n * sizeof(float));
    float* pr_2 = (float*)malloc(n * sizeof(float));
    void** vertex_dim = (void**)malloc(vertex_numsets * sizeof(void*));
    cudaDataType_t* vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets * sizeof(cudaDataType_t));
    nvgraphCSCTopology32I_t CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));

    // Initialize host data
    vertex_dim[0] = (void*)bookmark_h; vertex_dim[1] = (void*)pr_1, vertex_dim[2] = (void*)pr_2;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1] = CUDA_R_32F, vertex_dimT[2] = CUDA_R_32F;

    weights_h[0] = 0.333333f;
    weights_h[1] = 0.500000f;
    weights_h[2] = 0.333333f;
    weights_h[3] = 0.500000f;
    weights_h[4] = 0.500000f;
    weights_h[5] = 1.000000f;
    weights_h[6] = 0.333333f;
    weights_h[7] = 0.500000f;
    weights_h[8] = 0.500000f;
    weights_h[9] = 0.500000f;

    destination_offsets_h[0] = 0;
    destination_offsets_h[1] = 1;
    destination_offsets_h[2] = 3;
    destination_offsets_h[3] = 4;
    destination_offsets_h[4] = 6;
    destination_offsets_h[5] = 8;
    destination_offsets_h[6] = 10;

    source_indices_h[0] = 2;
    source_indices_h[1] = 0;
    source_indices_h[2] = 2;
    source_indices_h[3] = 0;
    source_indices_h[4] = 4;
    source_indices_h[5] = 5;
    source_indices_h[6] = 2;
    source_indices_h[7] = 3;
    source_indices_h[8] = 3;
    source_indices_h[9] = 4;

    bookmark_h[0] = 0.0f;
    bookmark_h[1] = 1.0f;
    bookmark_h[2] = 0.0f;
    bookmark_h[3] = 0.0f;
    bookmark_h[4] = 0.0f;
    bookmark_h[5] = 0.0f;

    // Starting nvgraph
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr(handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    int i;
    for (i = 0; i < 2; ++i)
    {
        check_status(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
    }
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    // First run with default values
    check_status(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));

    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
    printf("pr_1, alpha = 0.85\n"); for (i = 0; i < n; i++)  printf("%f\n", pr_1[i]); printf("\n");

    // Second run with different damping factor and an initial guess
    for (i = 0; i < n; i++)
        pr_2[i] = pr_1[i];

    nvgraphSetVertexData(handle, graph, vertex_dim[2], 2);
    check_status(nvgraphPagerank(handle, graph, 0, alpha2_p, 0, 1, 2, 0.0f, 0));

    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, vertex_dim[2], 2));
    printf("pr_2, alpha = 0.90\n"); for (i = 0; i < n; i++)  printf("%f\n", pr_2[i]); printf("\n");

    // Clean 
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(bookmark_h);
    free(pr_1);
    free(pr_2);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    donpp();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "  npp: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n\n\n";

    start = std::chrono::high_resolution_clock::now();
    donvgraph();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "   nv: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds\n";

    return 0;
}
