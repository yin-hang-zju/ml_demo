#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "BP.h"
int seed=1;Type MIN_ETA = 0.000001, BB = A / ( exp(-1.0/B/B) + 1.0);
//获取训练所有样本数据
void BP::GetData(const Vector<Data> _data)
{
    data = _data;
}

//开始进行训练
void BP::Train(bool debug /*=false */)
{
    printf("Begin to train BP NetWork!\n");
    GetNums();
    InitNetWork();
    int num = data.size();

    for(int iter = 0; iter <= ITERS; iter++)
    {
        if (debug)
            OutputNetwork(); //输出看看。可禁掉.. 
        for(int cnt = 0; cnt < num; cnt++)
        {
            //第一层输入节点赋值
            for(int i = 0; i < in_num; i++)
                x[0][i] = data.at(cnt).x[i];

            for(int one_sample_iter=0; one_sample_iter<ONEITER; one_sample_iter++)
            {
                ForwardTransfer();     
                if(GetError(cnt) < ERROR)    //如果误差比较小，则针对单个样本跳出循环
                    break;
                ReverseTransfer(cnt);  
            }
        }
        printf("This is the %d th trainning NetWork !\n", iter);

        Type accu = GetAccu();
        printf("All Samples Loss is %lf\n", accu);
        if(accu < ACCU) break;
        if(last_acc > 0 && accu > last_acc) { //误差震荡，需减少学习率 :
            if(ETA_W > MIN_ETA) {
                ETA_W *=   0.5;   //权值调整率
                ETA_B *=   0.5;    //阀值调整率
            } printf("eta_w = %lf, eta_b =%lf\n", ETA_W, ETA_B);
        }
        last_acc = accu;
    }
    printf("The BP NetWork train End!\n");
    if (debug)
            OutputNetwork(); //输出看看。可禁掉.. 
}

//根据训练好的网络来预测输出值
Vector<Type> BP::ForeCast(const Vector<Type> data)
{
    int n = data.size();
    assert(n == in_num);
    for(int i = 0; i < in_num; i++)
        x[0][i] = data[i];
    
    ForwardTransfer();
    Vector<Type> v;
    for(int i = 0; i < ou_num; i++)
        v.push_back(x[LAYER-1][i]);
    return v;
}

//输出训练出的网络结构
void BP::OutputNetwork()
{
  printf("{\n");
  //k=1:计算隐含层各个节点的输出值
    int k=1;
    for(int j = 0; j < hd_nums[1]; j++)
    {
        for(int i = 0; i < in_num; i++)
            printf("%lf ",w[k][i][j]);
        printf(", %lf\n", b[k][j]);
    }
    printf("\n");
    for(k=2; k<LAYER-1; k++) {
        for(int j = 0; j < hd_nums[k]; j++)
        {
            for(int i = 0; i < hd_nums[k-1]; i++)
                printf("%lf ",w[k][i][j]);
            printf(", %lf\n", b[k][j]);
        }
        printf("\n");
    }
  //k=2:计算输出层各节点的输出值
    k=LAYER-1;
    for(int j = 0; j < ou_num; j++)
    {
        for(int i = 0; i < hd_nums[k-1]; i++)
            printf("%lf ",w[k][i][j]);
        printf(", %lf\n", b[k][j]);
    }
  printf("}\n");
}
//获取网络节点数
void BP::GetNums()
{
    hd_nums[0] = in_num = data[0].x.size();                         //获取输入层节点数
    hd_nums[LAYER-1] = ou_num = data[0].y.size();                         //获取输出层节点数
    int hd = (int)sqrt((in_num + ou_num) * 1.0) + 5;   //获取隐含层节点数
    if(hd > NUM) hd = NUM;                     //隐含层数目不能超过最大设置
    for (int i=1; i<LAYER-1; i++) {
        //这个网上下的程序中w和b的[0]都没有意义,hd_num本只有一个(原来LAYER=3只有1个隐藏层)。为保持一致，改成数组后hd_num[0]也不使用.其实hd_num[0]应该保存in_num
        hd_nums[i] = hd;
    }
}

inline Type getRandNum() {
    return 1.0 - 2.0*rand()/RAND_MAX; //或者以[-r, r]上的均匀分布初始化某层的权值，其中：r=sqrt(6/(n_in+n_out))
}

//初始化网络
void BP::InitNetWork()
{

    ETA_W =   0.0035;   //权值调整率
    ETA_B =   0.001;    //阀值调整率
    last_acc = -1.0;    //上次模型的总误差..
    REGULAR  = 0.01;      //正则化Weight Decay
    A =       30.0   ;
    B =       10.0   ;//A和B是S型函数的参数
    ITERS =   5000    ;//最大训练次数,原来是1000
    ERROR =   0.002  ;//单个样本允许的误差
    ONEITER = 10000    ;//单个样本最大训练次数,原来没有上限
    ACCU =    0.005  ;//每次迭代允许的误差
    BB = A / ( exp(-1.0/B/B) + 1.0);
    memset(w, 0, sizeof(w));      //初始化权值和阀值为0，也可以初始化随机值
    memset(b, 0, sizeof(b));
    //return; 不应该初始化为0吧？0就死了..
    //seed = time(NULL); printf("seed = %d\n", seed); srand(seed); 
    int k;
    for(k=1; k<LAYER-1; k++) {
        for(int j = 0; j < hd_nums[k]; j++)
        {
            for(int i = 0; i < hd_nums[k-1]; i++)
                w[k][i][j] = getRandNum();
            b[k][j] = getRandNum();
        }
    }
    //k=2;
    for(int j = 0; j < ou_num; j++)
    {
        for(int i = 0; i < hd_nums[k-1]; i++)
            w[k][i][j] = getRandNum();
        b[k][j] = getRandNum();
    }

    //printf("in_num=%d, hd_num=%d, ou_num=%d\n", in_num, hd_num, ou_num);
    //OutputNetwork();
}

//工作信号正向传递子过程
void BP::ForwardTransfer()
{
    //计算隐含层各个节点的输出值..
    int k;
    for(k=1; k<LAYER-1; k++) {
        for(int j = 0; j < hd_nums[k]; j++)
        {
            Type t = 0;
            for(int i = 0; i < hd_nums[k-1]; i++)
                t += w[k][i][j] * x[k-1][i];
            t += b[k][j];
            x[k][j] = Activator(t);
        }
    }
    //计算输出层各节点的输出值
    for(int j = 0; j < ou_num; j++)
    {
        Type t = 0;
        for(int i = 0; i < hd_nums[k-1]; i++)
            t += w[k][i][j] * x[k-1][i];
        t += b[k][j];
        x[k][j] = Activator(t); //下载的代码如此。不过输出层的激活函数会不会用线性更好? TODO
    }
}

//计算单个样本的误差
Type BP::GetError(int cnt)
{
    Type ans = 0;
    for(int i = 0; i < ou_num; i++)
        ans += 0.5 * (x[LAYER-1][i] - data.at(cnt).y[i]) * (x[LAYER-1][i] - data.at(cnt).y[i]);
    return ans;
}

//误差信号反向传递子过程
void BP::ReverseTransfer(int cnt)
{   
    Type tmp[NUM+1], delta[LAYER]; 
    int k = LAYER-1;
    delta[k] = 0.0;
    for(int i = 0; i < hd_nums[k]; i++) {
        tmp[i] = x[k][i] - data.at(cnt).y[i]; 
        delta[k] += tmp[i] * Diff_Activator(x[k][i]);
    }

    for(k = LAYER-1; k>1; k--) { 
        for(int i = 0; i < hd_nums[k]; i++) {
            tmp[i] = delta[k+1] * w[k]; //tmp是矩阵，两个维度分别是相邻两层的神经元数量..
            Type delta = tmp * Diff_Activator(x[k][i]); tmp = delta * w //各向量矩阵的大小需要确认下 TODO
    }
} // tmp=(y-x[k]); delta = tmp multiply Diff_Activator(x[k]); tmp = delta * w
        //CalcDelta(cnt);   UpdateNetWork();
//计算所有样本的精度
Type BP::GetAccu()
{
    Type ans = 0;
    int num = data.size();
    for(int i = 0; i < num; i++)
    {
        int m = data.at(i).x.size();
        for(int j = 0; j < m; j++)
            x[0][j] = data.at(i).x[j];
        ForwardTransfer();
        int n = data.at(i).y.size();
        for(int j = 0; j < n; j++)
            ans += 0.5 * (x[LAYER-1][j] - data.at(i).y[j]) * (x[LAYER-1][j] - data.at(i).y[j]);
    }
    return ans / num;
}

//计算调整量
void BP::CalcDelta(int cnt)
{
    //计算输出层的delta
    int k = LAYER-1;
    for(int i = 0; i < ou_num; i++)
        d[k][i] = (x[k][i] - data.at(cnt).y[i]) * x[k][i] * (A - x[k][i]) / (A * B);
    while(k > 1) {
        //for(int i = 0; i < hd_nums[k]; i++)
        //    d[k][i] = (x[k][i] - data.at(cnt).y[i]) * x[k][i] * (A - x[k][i]) / (A * B);
        //计算隐含层的delta
        for(int i = 0; i < hd_nums[k-1]; i++)
        {
            Type t = 0;
            for(int j = 0; j < hd_nums[k]; j++)
                t += w[k][i][j] * d[k][j];
            d[k-1][i] = t * x[k-1][i] * (A - x[k-1][i]) / (A * B);
        }
        --k;
    }
}

//根据计算出的调整量对BP网络进行调整 251行
void BP::UpdateNetWork()
{
    //隐含层和输出层之间权值和阀值调整..
    int k = LAYER-1;
    for(int i = 0; i < hd_nums[k-1]; i++)
    {
        for(int j = 0; j < ou_num; j++)
            w[k][i][j] = w[k][i][j]*(1-ETA_W*REGULAR) - ETA_W * d[k][j] * x[k-1][i]; 
        //最后一层是不是不要正则化了 w[k][i][j] = w[k][i][j] - ETA_W * d[k][j] * x[k-1][i]; 
    }
    for(int i = 0; i < ou_num; i++)
        b[k][i] -= ETA_B * d[k][i];

    //输入层和隐含层们之间权值和阀值调整
    for(k-=1; k>0; k--) {
        for(int i = 0; i < hd_nums[k-1]; i++)
        {
            for(int j = 0; j < hd_nums[k]; j++)
                w[k][i][j] = w[k][i][j]*(1-ETA_W*REGULAR) - ETA_W * d[k][j] * x[k-1][i]; 
        }
        for(int i = 0; i < hd_nums[k]; i++)
            b[k][i] -= ETA_B * d[k][i];
    }

    return;
}

Type Sigmoid(const Type x)
{
    Type res =  A / (1 + exp(-x / B));
    return res;
}
//计算Activator函数的值
Type BP::Activator(const Type x)
{
    Type res =  A / (1 + exp(-x / B));
    return res;
}

Type Diff_Activator(const Type x) {
    Type t = Sigmoid(x);
    t = t * (1.0 - t) * BB;
    return t;
}

int main() {
    seed = time(NULL); 
    printf("seed = %d\n", seed); 
    srand(seed); 
    BP a;
    int i;
    /*************控制这里切换注释段,为啥不#ifdef..   */
    int n = 2;
    Vector<Type> x;
    //for(i=0; i<n; i++)
    x.push_back(0);
    x.push_back(1);
//{0.1,0.2,0.3};
    printf("x.size=%lu\n", x.size());
    Vector<Type> y;
    y.push_back(1);//0.6);
    Data d;
    d.x=x;
    d.y=y;
    Vector<Data> dataset;
    dataset.push_back(d);
    d.x[0] = 1;
    d.x[1] = 0;
    d.y[0] = 1;
    dataset.push_back(d);
    d.x[0] = 1;
    d.x[1] = 1;
    d.y[0] = 0;
    dataset.push_back(d);
    d.x[0] = 0;
    d.x[1] = 0;
    d.y[0] = 0;
    dataset.push_back(d);
    /*/
    int n = 3;
    Vector<Type> x;
    for(i=0; i<n; i++)
        x.push_back(0.1*(i+1));
//{0.1,0.2,0.3};
    printf("x.size=%lu\n", x.size());
    Vector<Type> y;
    y.push_back(1);//0.6);
    Data d;
    d.x=x;
    d.y=y;
    Vector<Data> dataset;
    dataset.push_back(d);
    for(i=0; i<n; i++)
        d.x[i] *= 2.0;
    //d.y[0] *= 2.0;
    dataset.push_back(d);
    
    for(int nd=0; nd<100; nd++) {
        d.y[0] = 0.0;
        for(i=0; i<n; i++) {
            d.x[i] = getRandNum(); //1.0 - 2.0*rand()/RAND_MAX;
            d.y[0] += d.x[i];
        }
        if (d.y[0] > 0)
            d.y[0] = 1;
        else
            d.y[0] = 0;
        dataset.push_back(d);
    }
/**/
    printf("in.size=%lu, %lu. sample num=%lu\n", dataset[0].x.size(), d.x.size(), dataset.size());

    a.GetData(dataset);
    a.Train(true);
    //Type acu = a.GetAccu();
    //printf("acu = %lf\n", acu);
    while(1) {
        printf("Please input x(%d):\n", n);
        for(i=0; i<n; i++) {
            Type input;
            if (scanf("%lf", &input) < 1)
                return 0;
            x[i] = input;
        }
        printf("res = %lf\n", a.ForeCast(x)[0]);
    }

    return 0;
}
