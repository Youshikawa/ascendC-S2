#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_Y>
class KernelXlogy {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelXlogy() {}
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                uint32_t totalLengthX1, uint32_t blockLengthX1, 
                                uint32_t totalLengthX2, uint32_t blockLengthX2, 
                                uint32_t totalLengthY, uint32_t blockLengthY,
                                uint32_t ALIGN_NUM, uint32_t tileLength) {
        /*
            内存属性初始化
        */
        this->pipe = pipeIn;
        this->ALIGN_NUM = ALIGN_NUM;
        this->tileLength = tileLength; // 总共一次可以计算的元素个数
        this->totalLengthX1 = totalLengthX1; // 总的需要计算的长度
        this->blockLengthX1 = blockLengthX1; // 需要多少个block（32字节为单位）
        this->totalLengthX2 = totalLengthX2; // 总的需要计算的长度
        this->blockLengthX2 = blockLengthX2; // 需要多少个block（32字节为单位）
        this->totalLengthY = totalLengthY;// Y的总长度
        this->blockLengthY = blockLengthY;// 需要多少block，对于y来说
        this->tileNum = totalLengthX1 / this->tileLength + (totalLengthX1 % this->tileLength ? 1 : 0);
        /*
            UB及GB分配
        */
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, blockLengthX1 + 128);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X1*)x2, blockLengthX2 + 128);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y , blockLengthY);
        this -> pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        this -> pipe->InitBuffer(Q_x2,BUFFER_NUM,this->tileLength * sizeof(TYPE_X1));
        this -> pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y)); 
        this -> pipe->InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        this -> pipe->InitBuffer(B_zeros, this->tileLength * sizeof(uint8_t));
        this -> pipe->InitBuffer(Term1, this->tileLength * sizeof(uint8_t));
        this -> pipe->InitBuffer(Term2, this->tileLength * sizeof(half));
        
        PreProcess();
    }
    __aicore__ inline void Process() {
        int32_t aligned = 128;
        if constexpr(std::is_same_v<T,float>) {
            aligned = 64;
        } 
        int32_t loopCount = this->tileNum;
        for (uint32_t i = 0; i < loopCount - 1; i += 1) {
            CopyIn(i,this->tileLength);
            Compute(i,this->tileLength, ((this->tileLength + aligned - 1) / aligned )* aligned);
            CopyOut(i,this->tileLength);
        }
        uint32_t finalLength = this -> blockLengthX1 -( this -> tileLength)*(loopCount - 1);
        uint32_t finalCacLength = this -> totalLengthX1 -( this -> tileLength)*(loopCount - 1);
        CopyIn(loopCount-1,finalLength);
        Compute(loopCount-1,finalCacLength,(finalCacLength + aligned - 1)/aligned * aligned);
        CopyOut(loopCount-1,finalLength);

    }
private://预处理
    __aicore__ inline void PreProcess() {

    } 
    
private://处理
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X1> xLocal2 = Q_x2.AllocTensor<TYPE_X1>();
        DataCopy(xLocal1, Gm_x1[progress*this->tileLength], length);
        DataCopy(xLocal2, Gm_x2[progress*this->tileLength], length);
        Q_x1.EnQue(xLocal1);
        Q_x2.EnQue(xLocal2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length, uint32_t CMPLength) {
        LocalTensor<TYPE_X1> xLocal1 = Q_x1.DeQue<TYPE_X1>(); 
        LocalTensor<TYPE_X1> xLocal2 = Q_x2.DeQue<TYPE_X1>(); 
        LocalTensor<TYPE_Y> yLocal = Q_y.AllocTensor<TYPE_Y>(); 
        if constexpr(std::is_same_v<T, half>){ 
            Ln(yLocal,xLocal2,length);
            Mul(yLocal,xLocal1,yLocal,length); 
            Q_x1.FreeTensor(xLocal1);
            Q_x2.FreeTensor(xLocal2);
            Q_y.EnQue(yLocal);
        } else { 
            Ln(yLocal,xLocal2,length);
            float sc1 = 0.0;
            auto zeros = B_zeros.Get<float>();
            Duplicate(zeros,sc1,CMPLength);
            auto term1 = Term1.Get<uint8_t>();
            Compare(term1,xLocal1,zeros,CMPMODE::NE,CMPLength);
            Mul(yLocal,xLocal1,yLocal,length); 
            // Select(yLocal,term1,yLocal,float(0.0),SELMODE::VSEL_TENSOR_SCALAR_MODE,length);
            
            Q_x1.FreeTensor(xLocal1);
            Q_x2.FreeTensor(xLocal2);
            Q_y.EnQue(yLocal);
        }
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress*this->tileLength],yLocal,length);
        Q_y.FreeTensor(yLocal);
    }
private://资源属性
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_zeros,B_bits, Term1,Term2;
    GlobalTensor<DTYPE_X1> Gm_x1;
    GlobalTensor<DTYPE_X1> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;  
    LocalTensor<TYPE_X1> zeros;
    LocalTensor<uint8_t> bits;
    uint32_t totalLengthX1;
    uint32_t blockLengthX1;
    uint32_t totalLengthX2;
    uint32_t blockLengthX2;
    uint32_t totalLengthY;
    uint32_t blockLengthY;
    uint32_t finalLoopLength;
    uint32_t finalLoopCacLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t ALIGN_NUM;

};
extern "C" __global__ __aicore__ void xlogy(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    TPipe pipe;
    KernelXlogy<DTYPE_X1, DTYPE_Y> op;
    op.Init(
        &pipe,x1,x2,y,
        tiling_data.totalLengthX1,tiling_data.blockLengthX1,
        tiling_data.totalLengthX2,tiling_data.blockLengthX2,
        tiling_data.totalLengthY,tiling_data.blockLengthY,
        tiling_data.ALIGN_NUM,tiling_data.block_size
    );
    op.Process();
}