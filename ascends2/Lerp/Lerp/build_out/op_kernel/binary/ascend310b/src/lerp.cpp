#include "kernel_operator.h" 
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X1, typename TYPE_X2, typename TYPE_X3, typename TYPE_Y> class Lerp {  //start end weight 
    using T = TYPE_X1;
public:
    __aicore__ inline Lerp() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); // 3
        this->tileLength = block_size; //5248
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); // 8

        auto startPointer = core_size * GetBlockIdx(); //0
        auto bufferlength = this->blockLength; // 
  

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferlength);
        Gm_x3.SetGlobalBuffer((__gm__ TYPE_X3*)x3 + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0); //1

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_x3, BUFFER_NUM, this->tileLength * sizeof(TYPE_X3));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr(std::is_same_v<T,half>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x3, this->tileLength * sizeof(float)); 
            pipe.InitBuffer(B_result, this->tileLength * sizeof(float));
        }
        
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        LocalTensor<TYPE_X3> x3 = Q_x3.AllocTensor<TYPE_X3>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);  
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        DataCopy(x3, Gm_x3[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
        Q_x3.EnQue(x3);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_X3> x3 = Q_x3.DeQue<TYPE_X3>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        if constexpr(std::is_same_v<T, half>) {
            auto fl_x1 = B_x1.Get<float>();
            auto fl_x2 = B_x2.Get<float>();
            auto fl_x3 = B_x3.Get<float>();
            auto res = B_result.Get<float>();
            Cast(fl_x1, x1, RoundMode::CAST_NONE, length);
            Cast(fl_x2, x2, RoundMode::CAST_NONE, length);
            Cast(fl_x3, x3, RoundMode::CAST_NONE, length); 
            Sub(res, fl_x2, fl_x1, length);
            Mul(res, fl_x3, res, length);
            Add(res, fl_x1, res, length);
            Cast(y, res, RoundMode::CAST_ROUND, length);
        } else {
            Sub(y, x2, x1, length);
            Mul(y, x3, y, length);
            Add(y, x1, y, length);
        } 
        
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_x3.FreeTensor(x3);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2, Q_x3;
    TBuf<QuePosition::VECCALC> B_x1, B_x2, B_x3, B_result;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y; 
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_X3> Gm_x3;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


template<typename TYPE_X1, typename TYPE_X2, typename TYPE_X3, typename TYPE_Y> class LerpBroadcast {
    using T = TYPE_X1;
public:
    __aicore__ inline LerpBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t* reduce1, uint32_t* reduce2, uint32_t* reduce3,uint32_t* shape, uint32_t dim) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); //3
        this->tileLength = block_size; //5248
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);//8
        this->reduce1 = reduce1; // [1]
        this->reduce2 = reduce2; // [1]
        this->reduce3 = reduce3; // [0]
        this->totalLength = totalLength; // 3
        this->ALIGN_NUM = ALIGN_NUM; // 8
        this->shape = shape; //[3,1]
        this->dim = dim; //2

        auto startPointer = core_size * GetBlockIdx(); //0
        auto bufferlength = this->blockLength; //8

        uint32_t inTotalLength1 = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce1[i] == 0){
                inTotalLength1 *= this->shape[i];
            }
        } // 1
        inTotalLength1 =(inTotalLength1 + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM; //8

        uint32_t inTotalLength2 = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce2[i] == 0){
                inTotalLength2 *= this->shape[i];
            }
        }//1
        inTotalLength2 =(inTotalLength2 + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;//8

        uint32_t inTotalLength3 = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce3[i] == 0){
                inTotalLength3 *= this->shape[i];
            }
        } //3
        inTotalLength3 =(inTotalLength3 + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM; //8

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, inTotalLength1);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, inTotalLength2);
        Gm_x3.SetGlobalBuffer((__gm__ TYPE_X3*)x3 + startPointer, inTotalLength3);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0); // 1

        pipe.InitBuffer(Q_x1, 1, this->tileLength * sizeof(TYPE_X1)); //
        pipe.InitBuffer(Q_x2, 1, this->tileLength * sizeof(TYPE_X2)); //
        pipe.InitBuffer(Q_x3, 1, this->tileLength * sizeof(TYPE_X3)); // 
        pipe.InitBuffer(Q_y, 1, this->tileLength * sizeof(TYPE_Y)); // 5248 * fp32 
        pipe.InitBuffer(B_tmp, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_tmp2, this->tileLength * sizeof(TYPE_Y));
        if constexpr(std::is_same_v<T,half>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x3, this->tileLength * sizeof(float)); 
            pipe.InitBuffer(B_result, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t count = this->totalLength / this->shape[this->dim - 1]; // 3 个 minVec 这里是一个元素 
        uint32_t totalLength = this->shape[this->dim - 1]; //minVec 一个元素 1
        this->tileNum = totalLength / this->tileLength + (totalLength % this->tileLength > 0); // 1次搬运就可完成计算
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        uint32_t dn3[21] = {0};
        auto dim = this -> dim - 1; //1
        d[dim] = dn1[dim] = dn2[dim] = dn3[dim] = 1; 
        for(int k = dim - 1; k >= 0; k --){
            d[k] = d[k + 1] * this->shape[k]; //d[0] = 3
            if(this->reduce1[k] == 0){ //reduce1[0] = 1
                dn1[k] = dn1[k + 1] * this->shape[k];
            }else{
                dn1[k] = dn1[k + 1]; // dn1[0] = 1
            }
            if(this->reduce2[k] == 0){ //reduce2[0] = 1
                dn2[k] = dn2[k + 1] * this->shape[k];
            }else{
                dn2[k] = dn2[k + 1]; //dn2[0] = 1
            }
            if(this->reduce3[k] == 0){ //reduce3[0] = 0
                dn3[k] = dn3[k + 1] * this->shape[k]; //dn3[0] = 3
            }else{
                dn3[k] = dn3[k + 1];
            }
        }
        


        for(int j=0;j<count;j++){ // 3
            uint32_t start1 = 0, start2 = 0, start3 = 0; 
            for(int k=dim-1;k>=0;k--){ // 只跑一次
                if(this->reduce1[k] == 0){
                    start1 += dn1[k + 1] * ((j / d[k + 1]) % this->shape[k]);
                }
                if(this->reduce2[k] == 0){
                    start2 += dn2[k + 1] * ((j / d[k + 1]) % this->shape[k]);
                }
                if(this->reduce3[k] == 0){
                    start3 += dn3[k + 1] * ((j / d[k + 1]) % this->shape[k]);
                }
            } // start3 = 1
            int32_t loopCount = this->tileNum; // 1
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyIn(start1 * totalLength, start2 * totalLength, start3 * totalLength, i, this->tileLength);
                Compute(i, this->tileLength);
                CopyOut(j * totalLength, i, this->tileLength);
            }
            uint32_t length = totalLength - this->tileLength * (loopCount - 1); // 3
            auto length_align = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM; // 8
            CopyIn(start1 * totalLength, start2 * totalLength, start3 * totalLength, loopCount - 1, length_align);
            Compute(loopCount - 1, length);
            CopyOut(j * totalLength, loopCount - 1, (length + 31) / 32 * 32);
        }
        
        
    }
private:
    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2, uint32_t start3, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        LocalTensor<TYPE_X3> x3 = Q_x3.AllocTensor<TYPE_X3>();
        DataCopy(x1, Gm_x1[start1 + progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[start2 + progress * this->tileLength], length);
        DataCopy(x3, Gm_x3[start3 + progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
        Q_x3.EnQue(x3);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_X3> x3 = Q_x3.DeQue<TYPE_X3>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        
        if constexpr(std::is_same_v<T, half>) {
            auto fl_x1 = B_x1.Get<float>();
            auto fl_x2 = B_x2.Get<float>();
            auto fl_x3 = B_x3.Get<float>();
            auto res = B_result.Get<float>();
            Cast(fl_x1, x1, RoundMode::CAST_NONE, length);
            Cast(fl_x2, x2, RoundMode::CAST_NONE, length);
            Cast(fl_x3, x3, RoundMode::CAST_NONE, length); 
            Sub(res, fl_x2, fl_x1, length);
            Mul(res, fl_x3, res, length);
            Add(res, fl_x1, res, length);
            Cast(y, res, RoundMode::CAST_ROUND, length);
        } else { 
            Sub(y, x2, x1, length);     
            Mul(y, x3, y, length); 
            Add(y, x1, y, length); 
        } 

        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_x3.FreeTensor(x3);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(uint32_t start, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[start + progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> Q_x1, Q_x2, Q_x3;
    TBuf<QuePosition::VECCALC> B_x1, B_x2, B_x3, B_result, B_tmp, B_tmp2;
    TQue<QuePosition::VECOUT, 1> Q_y; 
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_X3> Gm_x3;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t* reduce1;
    uint32_t* reduce2;
    uint32_t* reduce3;
    uint32_t* shape;
    uint32_t dim;
    uint32_t ALIGN_NUM;
};

extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(1)){
        Lerp<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        LerpBroadcast<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size,tiling_data.core_remain, 
            tiling_data.reduce1, tiling_data.reduce2, tiling_data.reduce3, tiling_data.shape, tiling_data.dim);
        op.Process();
    }
}