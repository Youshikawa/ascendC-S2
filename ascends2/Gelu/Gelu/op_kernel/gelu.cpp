#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_Y>
class KernelGelu {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t ALIGN_NUM, uint32_t block_size,
                                uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->ALIGN_NUM = ALIGN_NUM;
 
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ TYPE_X1*)x , bufferlength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y , bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        if constexpr (std::is_same_v<T,half>) {
            pipe.InitBuffer(B_key1_tmpBuffer, this->tileLength * sizeof(TYPE_X1));
        } else {
            pipe.InitBuffer(B_ori_x, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(B_bits_2, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(B_zeros, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_pow, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_tmpBuffer2, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_ones, this->tileLength * sizeof(float));
            this->ones = B_ones.Get<float>();
            this->zeros = B_zeros.Get<float>();
            this->bits = B_bits.Get<uint8_t>();
            this->bits_2 = B_bits_2.Get<uint8_t>();
            Duplicate(this->ones, float(1.0), this->tileLength);
            Duplicate(this->zeros, float(0), this->tileLength);
            pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
static constexpr float root2_rec = 0.70710678118654752440084436210485;
/*
    static constexpr float p1 = 0.0705230784;
    static constexpr float p2 = 0.0422820123;
    static constexpr float p3 = 0.0092705272;
    static constexpr float p4 = 0.0001520143;
    static constexpr float p5 = 0.0002765672;
    static constexpr float p6 = 0.0000430638;
    static constexpr uint32_t erfIndex = 16;
    // erf 实现思路：
    //     因为计算的是张量所以 无法对具体 +-x特判处理 但erf是奇函数 erf(x) = -erf(-x)
    //     可以先将 tensor 中 abs处理 并保存负数位置 接着 erf计算，并保存两份，一份+erf一份-erf接着 根据负数位置Select即可
    //***************精度不够***********
    //不使用模板 默认对float32处理保证高精度
    //cmpLength 一定保证256 字节对齐因为使用了 Compare
    __aicore__ inline void  ErfCustom(LocalTensor<float>& x, LocalTensor<float>& y, LocalTensor<float>& workBuffer, LocalTensor<float>& x_pow, uint32_t length, uint32_t cmpLength = 0) { //将结果保存至y中
        if (cmpLength == 0) cmpLength = length;
        Compare(this->bits, x, this->zeros, CMPMODE::LT, cmpLength); // 1 为 x < 0 
        Compare(this->bits_2, x, this->zeros, CMPMODE::GE, cmpLength); // 1 为 x >= 0 
        Abs(x, x, length);

        Muls(workBuffer, x, p1, length); // p1*x
        Adds(y, workBuffer, float(1.0), length); // y = 1 + p1*x 

        Mul(x_pow, x, x, length); // x**2
        Muls(workBuffer, x_pow, p2, length); // p2*x**2
        Add(y, y, workBuffer, length); // y += p2*x**2

        Mul(x_pow, x_pow, x, length); // x ** 3
        Muls(workBuffer, x_pow, p3, length);
        Add(y, y, workBuffer, length);
        
        Mul(x_pow, x_pow, x, length);
        Muls(workBuffer, x_pow, p4, length);
        Add(y, y, workBuffer, length);
        
        Mul(x_pow, x_pow, x, length);
        Muls(workBuffer, x_pow, p5, length);
        Add(y, y, workBuffer, length);
        
        Mul(x_pow, x_pow, x, length);
        Muls(workBuffer, x_pow, p6, length);
        Add(y, y, workBuffer, length);

        //这里针对 n = 16， 就不在写泛化的ksm了
        Mul(y, y, y, length); // y2
        Mul(y, y, y, length); // y4
        Mul(y, y, y, length); // y8
        Mul(y, y, y, length); // y16

        Div(y, this->ones, y, length);

        Duplicate(workBuffer, float(1.0), length);
        Sub(x, workBuffer, y, length); // erf down

        Muls(workBuffer, x, float(-1.0), length); // -erf
        Select(y, this->bits, workBuffer, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length); // y = 0.0 -erf 0.0 
        Select(workBuffer, this->bits_2, x, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Add(y, workBuffer, y, length);
    }

*/
/*
    //这里 函数误差太大设计Reciprocal 除法， 但是 t城的打太多不行 或者 优化 Reciprocal
    static constexpr float p1 = -1.26551223;
    static constexpr float p2 = 1.00002368;
    static constexpr float p3 = 0.37409196;
    static constexpr float p4 = 0.09678418;
    static constexpr float p5 = -0.18628806;
    static constexpr float p6 = 0.27886807;
    static constexpr float p7 = -1.13520398;
    static constexpr float p8 = 1.48851587;
    static constexpr float p9 = -0.82215223;
    static constexpr float p10 = 0.17087277;
    
    __aicore__ inline void ErfCustomDelt(LocalTensor<float>& x, LocalTensor<float>& y, LocalTensor<float>& workBuffer, LocalTensor<float>& t_pow, uint32_t length) {
        Abs(x, x, length);

        Mul(y, x, x, length);
        Muls(y, y, float(-1.0), length);

        Muls(x, x, float(0.5), length);
        Adds(x, x, float(1.0), length);
        Reciprocal(x, x, length); // t
        Duplicate(x, float(0.8), length);


        Adds(t_pow, x, float(0.0), length); // t_pow

        Adds(y, y, p1, length); // -x2 - p1

        Muls(workBuffer, t_pow, p2, length); // t * p2 
        Add(y, y, workBuffer, length);

        Mul(t_pow, t_pow, x, length); //t2
        Muls(workBuffer, t_pow, p3, length);
        Add(y, y, workBuffer, length);

        Mul(t_pow, t_pow, x, length); //t3
        Muls(workBuffer, t_pow, p4, length);
        Add(y, y, workBuffer, length);
        
        Mul(t_pow, t_pow, x, length); //t4
        Muls(workBuffer, t_pow, p5, length);
        Add(y, y, workBuffer, length);
        
        Mul(t_pow, t_pow, x, length); //t5
        Muls(workBuffer, t_pow, p6, length);
        Add(y, y, workBuffer, length);

        Mul(t_pow, t_pow, x, length); //t6
        Muls(workBuffer, t_pow, p7, length);
        Add(y, y, workBuffer, length);

        
        Mul(t_pow, t_pow, x, length); //t7
        Muls(workBuffer, t_pow, p8, length);
        Add(y, y, workBuffer, length);
        
        Mul(t_pow, t_pow, x, length); //t8
        Muls(workBuffer, t_pow, p9, length);
        Add(y, y, workBuffer, length);
        
        Mul(t_pow, t_pow, x, length); //t9
        Muls(workBuffer, t_pow, p10, length);
        Add(y, y, workBuffer, length);
         
        Exp(y, y, length);
        Mul(y, x, y, length);
    }

    __aicore__ inline void ErfCustom(LocalTensor<float>& x, LocalTensor<float>& y, LocalTensor<float>& workBuffer, LocalTensor<float>& workBuffer2, uint32_t length, uint32_t cmpLength = 0) {
        if (cmpLength == 0) cmpLength = length;
        Compare(this->bits, x, this->zeros, CMPMODE::LT, cmpLength); // 1 为 x < 0 
        Compare(this->bits_2, x, this->zeros, CMPMODE::GE, cmpLength); // 1 为 x >= 0 
        ErfCustomDelt(x,y,workBuffer,workBuffer2,length); 
        
        Adds(workBuffer, y, float(-1.0), length); // T - 1  ...  x < 0

        Muls(workBuffer2, workBuffer, float(-1.0), length); // - (T - 1)  =  1 - T  ...  x >= 0

        Select(x, this->bits, workBuffer, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        Select(y, this->bits_2, workBuffer2, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Add(y, y, x, length);
    }
*/
/*
    // 同样reciprocal 误差太大
    static constexpr float p = 0.3275911;
    static constexpr float a1 = 0.254829592;
    static constexpr float a2 = -0.284496736;
    static constexpr float a3 = 1.421413741;
    static constexpr float a4 = -1.453152027;
    static constexpr float a5 = 1.061405429;

    __aicore__ inline void ErfCustom(LocalTensor<float>& x, LocalTensor<float>& y, LocalTensor<float>& workBuffer, LocalTensor<float>& workBuffer2, uint32_t length, uint32_t cmpLength = 0) {
        if (cmpLength == 0) cmpLength = length;
        Compare(this->bits, x, this->zeros, CMPMODE::LT, cmpLength); // 1 为 x < 0 
        Compare(this->bits_2, x, this->zeros, CMPMODE::GE, cmpLength); // 1 为 x >= 0 
        
        Abs(x, x, length);

        auto workBuffer3 = B_tmpBuffer2.Get<float>();

        Mul(y, x, x, length);  // y = x2
        Muls(y, y, float(-1.0), length); // y = -x2
        Exp(y, y, length); // y = exp(-x2)

        Muls(x, x, p, length);
        Adds(x, x, float(1.0), length);
        Div(x, this->ones, x, length);  // t
        
        Adds(workBuffer, x, float(0.0), length);

        Muls(workBuffer2, x, a1, length); 

        Mul(workBuffer, workBuffer, x, length); 
        Muls(workBuffer3, workBuffer, a2, length);
        Add(workBuffer2, workBuffer2, workBuffer3, length);
        
        Mul(workBuffer, workBuffer, x, length);
        Muls(workBuffer3, workBuffer, a3, length);
        Add(workBuffer2, workBuffer2, workBuffer3, length);

        Mul(workBuffer, workBuffer, x, length);
        Muls(workBuffer3, workBuffer, a4, length);
        Add(workBuffer2, workBuffer2, workBuffer3, length);

        Mul(workBuffer, workBuffer, x, length);
        Muls(workBuffer3, workBuffer, a5, length);
        Add(workBuffer2, workBuffer2, workBuffer3, length);

        Mul(y, y, workBuffer2, length);
        Muls(y, y, float(-1.0), length);
        Adds(y, y, float(1.0), length); // erf
        
        Muls(workBuffer, y, float(-1.0), length); // -erf

        Select(x, this->bits, workBuffer, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        Select(workBuffer2, this->bits_2, y, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Add(y, workBuffer2, x, length);
    }

*/
    static constexpr float p =  0.47047;
    static constexpr float a1 = 0.3480242;
    static constexpr float a2 = -0.0958798;
    static constexpr float a3 = 0.7478556;

    __aicore__ inline void ErfCustom(LocalTensor<float>& x, LocalTensor<float>& y, LocalTensor<float>& workBuffer, LocalTensor<float>& workBuffer2, uint32_t length, uint32_t cmpLength = 0) {
        if (cmpLength == 0) cmpLength = length;
        Compare(this->bits, x, this->zeros, CMPMODE::LT, cmpLength); // 1 为 x < 0 
        Compare(this->bits_2, x, this->zeros, CMPMODE::GE, cmpLength); // 1 为 x >= 0 
        auto workBuffer3 = B_tmpBuffer2.Get<float>();

        Abs(x, x, length);
        
        // Mul(workBuffer, x, x, length);
        // Muls(workBuffer, workBuffer, float(-1.0), length);
        // Exp(y, workBuffer, length);

        // Muls(workBuffer, x, p, length);
        // Adds(workBuffer, workBuffer, float(1.0), length);
        // Div(workBuffer, this->ones, workBuffer, length); // t
        // Adds(workBuffer2, workBuffer, float(0.0), length); // t_pow
        
        // Muls(x, workBuffer2, a1, length);
        
        // Mul(workBuffer2, workBuffer2, workBuffer, length);
        // Muls(workBuffer3, workBuffer2, a2, length);
        // Add(x, x, workBuffer3, length);

        // Mul(workBuffer2, workBuffer2, workBuffer, length);
        // Muls(workBuffer3, workBuffer2, a3, length);
        // Add(x, x, workBuffer3, length);

        // Mul(y, y, x, length);
        // Muls(y, y, float(-1.0), length);
        float b1 = 0.57722, b2 = 0.32759; 

        Adds(y, y, float(1.0), length); //erf

        Muls(workBuffer, y, float(-1.0), length); // -erf

        Select(x, this->bits, workBuffer, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);  
        Select(workBuffer2, this->bits_2, y, float(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Add(y, workBuffer2, x, length);
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal = inQueueX.AllocTensor<TYPE_X1>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal = inQueueX.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr(std::is_same_v<T,half>) {
            auto tmp = B_key1_tmpBuffer.Get<TYPE_X1>();
 
            TYPE_X1 c1 = 0.5, c2 = 0.79788456080286535587989211986876, c3 = 0.044715, c4 = 1.0, c5 = -1.0;

 
            Mul(tmp, xLocal, xLocal, length); // x * x
            Muls(tmp, tmp, c3, length); // c3 * x * x
            Adds(tmp, tmp, c4, length); // 1 + c3 * x * x
            Mul(tmp, tmp, xLocal, length); // x * (1 + c3 * x * x)
            Muls(tmp, tmp, c2, length); // b saved in tmp
            Add(tmp, tmp, tmp, length); // 2b saved in tmp

            Muls(yLocal, xLocal, c1, length); //c1 * x saved in yLocal


            Exp(tmp, tmp, length); // e2b saved in tmp
            Adds(xLocal, tmp, c4, length);
            Adds(tmp, tmp, c5, length);
            Div(tmp, tmp, xLocal, length); // tanh

            Adds(tmp, tmp, c4, length);
            Mul(yLocal, yLocal, tmp, length);
 

        } else {
            auto x_pow = B_pow.Get<float>();
            auto tmp = tmpBuffer.Get<float>();
            auto ori_x = B_ori_x.Get<float>();
            Adds(ori_x, xLocal, float(0.0), length);
            Duplicate(tmp, root2_rec, length);
            Mul(xLocal, xLocal, tmp, length);
            ErfCustom(xLocal, yLocal, tmp, x_pow, length, (length + (this->ALIGN_NUM * 8) - 1) / (this->ALIGN_NUM * 8) * (this->ALIGN_NUM * 8));
            Adds(yLocal, yLocal, float(1.0), length);
            Muls(yLocal, yLocal, float(0.5), length);
            Mul(yLocal, yLocal, ori_x, length);
  
        }
        // length * AN = 多少个

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
 
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> B_key1_tmpBuffer, tmpBuffer, B_bits, B_bits_2, B_zeros, B_pow, B_ori_x, B_tmpBuffer2, B_ones;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    LocalTensor<uint8_t> bits, bits_2;
    LocalTensor<float> zeros,ones;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t ALIGN_NUM;

};

 

extern "C" __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
 
    KernelGelu<DTYPE_X,DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
 
}