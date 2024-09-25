#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class ThreeNN { 
public:
    __aicore__ inline ThreeNN() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, TPipe* pipeIn,
                                uint64_t* tileMask, uint64_t* finalMask,
                                uint32_t tilePointCount, uint32_t finalLoopPointCount,
                                uint32_t totalLength_bn3, uint32_t blockLength_bn3, uint32_t totalLength_bm3, uint32_t blockLength_bm3,
                                uint32_t batchLength_bn3, uint32_t batchLength_bm3, uint32_t batchCount, uint32_t nCount,
                                uint32_t loopCount, uint32_t finalLoopLength, uint32_t finalLoopCacLength, uint32_t ALIGN_NUM, uint32_t block_size) {
        this->pipe = pipeIn;

        this->tilePointCount = tilePointCount;
        this->finalLoopPointCount = finalLoopPointCount;
        
        this->totalLength_bn3 = totalLength_bn3;
        this->blockLength_bn3 = blockLength_bn3;

        this->totalLength_bm3 = totalLength_bm3;
        this->blockLength_bm3 = blockLength_bm3;

        this->batchLength_bn3 = batchLength_bn3;
        this->batchLength_bm3 = batchLength_bm3;

        this->batchCount = batchCount;
        this->nCount = nCount;

        this->loopCount = loopCount;
        this->finalLoopLength = finalLoopLength;
        this->finalLoopCacLength = finalLoopCacLength;

        this->ALIGN_NUM = ALIGN_NUM;
        this->tileLength = block_size;

        this->tileMask = tileMask; 
        this->finalMask = finalMask; 

        Gm_x1.SetGlobalBuffer((__gm__ float*)x1, this->blockLength_bn3);
        Gm_x2.SetGlobalBuffer((__gm__ float*)x2, this->blockLength_bm3);
        Gm_y1.SetGlobalBuffer((__gm__ float*)y1, this->blockLength_bn3);
        Gm_y2.SetGlobalBuffer((__gm__ int32_t*)y2, this->blockLength_bn3);

        pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(Q_x2_x, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(Q_x2_y, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(Q_x2_z, BUFFER_NUM, this->tileLength * sizeof(float)); 
        pipe->InitBuffer(Q_y1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(Q_y2, BUFFER_NUM, this->tileLength * sizeof(uint32_t)); 
        
  
        pipe->InitBuffer(B_min1, 8 * sizeof(float)); 
        pipe->InitBuffer(B_min2, 8 * sizeof(float)); 
        pipe->InitBuffer(B_min3, 8 * sizeof(float));  
        pipe->InitBuffer(B_bits, 32 * sizeof(uint8_t)); 
        
        this->min1 = B_min1.Get<float>();
        this->min2 = B_min2.Get<float>();
        this->min3 = B_min3.Get<float>();
        this->bits = B_bits.Get<uint8_t>();
    }
    __aicore__ inline void Process() {
        int32_t start_x1, start_x2;

        for (int32_t i = 0; i < batchCount; i ++) { //2
            start_x1 = i * batchLength_bn3; //0
            start_x2 = i * batchLength_bm3; //0
            for (int32_t j = 0; j < nCount; j ++) { // 1 
                idx1 = 0;
                idx2 = 0;
                idx3 = 0;
                Duplicate(min1,float(1e40),8);
                Duplicate(min2,float(1e40),8);
                Duplicate(min3,float(1e40),8);
                for (int32_t k = 0; k < loopCount - 1; k ++) { 
                    CopyIn(start_x1, start_x2, k, tileLength);
                    Compute(k, tileLength, tilePointCount, tileMask, false);
                    CopyOut(start_x1, 0);
                    start_x2 += 63;
                }
                CopyIn(start_x1, start_x2, loopCount - 1, finalLoopLength); //0, 0, 0, 8, 1
                Compute(loopCount - 1, finalLoopCacLength, finalLoopPointCount, finalMask,true); // 0, 8, 1, 1,
                CopyOut(start_x1, 8, true); //3个自动向上对齐为8个 //0, 1
                start_x1 += 3;
            }
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t start_x1, int32_t start_x2, int progress, uint32_t length) {
        LocalTensor<float> x1 = Q_x1.AllocTensor<float>();
        LocalTensor<float> x2_x = Q_x2_x.AllocTensor<float>();
        LocalTensor<float> x2_y = Q_x2_y.AllocTensor<float>();
        LocalTensor<float> x2_z = Q_x2_z.AllocTensor<float>();
        
        DataCopy(x1, Gm_x1[start_x1], 8);
        DataCopy(x2_x, Gm_x2[start_x2], length);
        DataCopy(x2_y, Gm_x2[start_x2 + 1], length);
        DataCopy(x2_z, Gm_x2[start_x2 + 2], length);
        Q_x1.EnQue(x1);
        Q_x2_x.EnQue(x2_x);
        Q_x2_y.EnQue(x2_y);
        Q_x2_z.EnQue(x2_z);
    }
    __aicore__ inline void Compute(int progress, uint32_t length, uint32_t pointCount, uint64_t* mask, bool isLast) {
        LocalTensor<float> x1 = Q_x1.DeQue<float>();
        LocalTensor<float> x2_x = Q_x2_x.DeQue<float>();
        LocalTensor<float> x2_y = Q_x2_y.DeQue<float>();
        LocalTensor<float> x2_z = Q_x2_z.DeQue<float>();
        LocalTensor<float> y1 = Q_y1.AllocTensor<float>();
        LocalTensor<int32_t> y2 = Q_y2.AllocTensor<int32_t>();

        Muls(x1, x1, float(-1), 3); 

        Adds(x2_x, x2_x, x1.GetValue(0), mask, 1, { 1, 1, 8, 8 });
        Adds(x2_y, x2_y, x1.GetValue(1), mask, 1, { 1, 1, 8, 8 });
        Adds(x2_z, x2_z, x1.GetValue(2), mask, 1, { 1, 1, 8, 8 });

        Mul(x2_x, x2_x, x2_x, mask, 1, { 1, 1, 1, 8, 8, 8 });
        Mul(x2_y, x2_y, x2_y, mask, 1, { 1, 1, 1, 8, 8, 8 });
        Mul(x2_z, x2_z, x2_z, mask, 1, { 1, 1, 1, 8, 8, 8 });

        Add(x2_x, x2_x, x2_y, mask, 1, { 1, 1, 1, 8, 8, 8 });
        Add(x2_x, x2_x, x2_z, mask, 1, { 1, 1, 1, 8, 8, 8 });

        ReduceMin(x2_y, x2_x, x2_z, mask, 1, 8, true);
        float c_min1 = x2_y.GetValue(0);
        uint32_t min1_pos = x2_y.ReinterpretCast<uint32_t>().GetValue(1);
        x2_x.SetValue(min1_pos, float(1e40));

        ReduceMin(x2_y, x2_x, x2_z, mask, 1, 8, true);
        float c_min2 = x2_y.GetValue(0);
        uint32_t min2_pos = x2_y.ReinterpretCast<uint32_t>().GetValue(1); 
        x2_x.SetValue(min2_pos, float(1e40));

        ReduceMin(x2_y, x2_x, x2_z, mask, 1, 8, true);
        float c_min3 = x2_y.GetValue(0);
        uint32_t min3_pos = x2_y.ReinterpretCast<uint32_t>().GetValue(1);
        
        

        if (isLast) {
            y1.SetValue(0,min1.GetValue(0));
            y1.SetValue(1,min2.GetValue(0));
            y1.SetValue(2,min3.GetValue(0));

            y2.SetValue(0,idx1);
            y2.SetValue(1,idx2);
            y2.SetValue(2,idx3);
        }
        
        Q_y1.EnQue(y1);
        Q_y2.EnQue(y2);
        Q_x1.FreeTensor(x1);
        Q_x2_x.FreeTensor(x2_x);
        Q_x2_y.FreeTensor(x2_y);
        Q_x2_z.FreeTensor(x2_z);
    }
    __aicore__ inline void CopyOut(int32_t start_x1, int progress, bool reallyCopyOut = false) {
        LocalTensor<float> y1 = Q_y1.DeQue<float>();
        LocalTensor<int32_t> y2 = Q_y2.DeQue<int32_t>();
        if (reallyCopyOut) {
            DataCopy(Gm_y1[start_x1], y1, 8);
            DataCopy(Gm_y2[start_x1], y2, 8);
        }
        Q_y1.FreeTensor(y1);
        Q_y2.FreeTensor(y2);
    }
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2_x, Q_x2_y, Q_x2_z;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y1, Q_y2;  
    TBuf<QuePosition::VECCALC> B_min1, B_min2, B_min3, B_bits;
    GlobalTensor<float> Gm_x1, Gm_x2, Gm_y1;
    GlobalTensor<int32_t> Gm_y2;
    LocalTensor<float> min1, min2, min3;
    LocalTensor<uint8_t> bits;
    int32_t idx1 = -1, idx2 = -1, idx3 = -1;

    uint32_t tilePointCount;
    uint32_t finalLoopPointCount;

    uint32_t totalLength_bn3;
    uint32_t blockLength_bn3;

    uint32_t totalLength_bm3;
    uint32_t blockLength_bm3;
    
    uint32_t batchLength_bn3;
    uint32_t batchLength_bm3;
    
    uint32_t batchCount;
    uint32_t nCount;
    
    uint32_t loopCount;
    uint32_t finalLoopLength;
    uint32_t finalLoopCacLength;
    
    uint32_t ALIGN_NUM;
    uint32_t tileLength;

    uint64_t* tileMask;
    uint64_t* finalMask; 
};



extern "C" __global__ __aicore__ void three_nn(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    /*
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2,
        uint32_t totalLength_bn3, uint32_t blockLength_bn3, uint32_t totalLength_bm3, uint32_t blockLength_bm3,
        uint32_t batchLength_bn3, uint32_t batchLength_bm3, uint32_t batchCount, uint32_t nCount,
        uint32_t loopCount, uint32_t finalLoopLength, uint32_t finalLoopCacLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t aivNum
    */
    TPipe pipe;
    ThreeNN op;
    op.Init(xyz1, xyz2, dist, indices, &pipe,
        tiling_data.tileMask, tiling_data.finalMask,
        tiling_data.tilePointCount, tiling_data.finalLoopPointCount,
        tiling_data.totalLength_bn3,tiling_data.blockLength_bn3,tiling_data.totalLength_bm3,tiling_data.blockLength_bm3,
        tiling_data.batchLength_bn3,tiling_data.batchLength_bm3,tiling_data.batchCount,tiling_data.nCount,
        tiling_data.loopCount,tiling_data.finalLoopLength,tiling_data.finalLoopCacLength,tiling_data.ALIGN_NUM,tiling_data.block_size);
    op.Process();
}