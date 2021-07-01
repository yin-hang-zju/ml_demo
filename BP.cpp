#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "BP.h"

int seed=1;
Type MIN_ETA = 1E-99, SMALL_ETA=0.0001, TOO_SMALL=1E-50, BB;// = A / ( exp(-1.0/B/B) + 1.0);
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
    int min_iter = -1;
    Type min_acc = 99999, last_error = -1;
    for(int iter = 0; iter <= ITERS; iter++)
    {
        if (debug)
            OutputNetwork(); //输出看看。可禁掉.. 
        //int cnt = rand()%num; //每个样本循环时，并没有算平均的梯度，所以与随机梯度下降应该一样?反而有固定顺序的缺点..
        for(int cnt = 0; cnt < num; cnt++)
        {
            //第一层输入节点赋值
            for(int i = 0; i < in_num; i++)
                x[0][i] = data.at(cnt).x[i];
            for(int one_sample_iter=0; one_sample_iter<ONEITER; one_sample_iter++)
            {
                ForwardTransfer(); 
                Type error = GetError(cnt);    
                if(error < ERROR) 
                    break; //如果误差比较小，则针对单个样本跳出循环
                AdjustEta(last_error, error); 
                last_error = error;
                ReverseTransfer(cnt);  
            }
        }
        printf("This is the %d th trainning NetWork !\n", iter);

        Type accu = GetAccu();
        printf("All Samples Loss is %.22lg\n", accu);
        if(accu < ACCU) {
            if(accu < ACCU) //可以debug时手工禁止跳出..
                break;
        }
        //误差震荡，需减少学习率 
        if(AdjustEta(last_acc, accu) == false) {
            if (ETA_W < SMALL_ETA) { //若未震荡，可以让学习率恢复一下..
                ETA_W = SMALL_ETA; 
                ETA_B = 0.1*SMALL_ETA; 
            }
        }
        last_acc = accu; 
        if (accu < min_acc) { //只记录最小值..
            min_acc = accu;
            min_iter = iter;
        }
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
            printf("%lg ",w[k][i][j]);
        printf(", %lg\n", b[k][j]);
    }
    printf("\n");
    for(k=2; k<LAYER-1; k++) {
        for(int j = 0; j < hd_nums[k]; j++)
        {
            for(int i = 0; i < hd_nums[k-1]; i++)
                printf("%lg ",w[k][i][j]);
            printf(", %lg\n", b[k][j]);
        }
        printf("\n");
    }
  //k=2:计算输出层各节点的输出值
    k=LAYER-1;
    for(int j = 0; j < ou_num; j++)
    {
        for(int i = 0; i < hd_nums[k-1]; i++)
            printf("%lg ",w[k][i][j]);
        printf(", %lg\n", b[k][j]);
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

inline Type getRandNum(int in, int out) {
    Type para = 1.0;
    if (in > 0 && out > 0) {
        para = sqrt(12.0/(in+out));
    }
    return para - para*2.0*rand()/RAND_MAX; //或者以[-r, r]上的均匀分布初始化某层的权值，其中：r=sqrt(6/(n_in+n_out))
}

//初始化网络
void BP::InitNetWork()
{

    ETA_W =   0.0035;   //权值调整率
    ETA_B =   0.001;    //阀值调整率
    last_acc = -1.0;    //上次模型的总误差..
    REGULAR  = 0; //一开始可不加,眼看过拟合了再加。0.01;      //正则化Weight Decay
    A =  1.0;//     30.0   ;
    B =  0.1;//     10.0   ;//A和B是S型函数的参数
    ITERS =   999000    ;//最大训练次数,原来是1000
    ERROR =   0.002  ;//单个样本允许的误差
    ONEITER = 10000    ;//单个样本最大训练次数,原来没有上限
    ACCU =    0.00000001; //0.005  ;//每次迭代允许的误差
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
                w[k][i][j] = getRandNum(hd_nums[k-1], hd_nums[k]);
            b[k][j] = getRandNum(0, 0);
        }
    }
    //k=LAYER-1;
    for(int j = 0; j < ou_num; j++)
    {
        for(int i = 0; i < hd_nums[k-1]; i++)
            w[k][i][j] = getRandNum(hd_nums[k-1], hd_nums[k]);
        b[k][j] = getRandNum(0, 0);
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
    } //k=LAYER-1;
    //计算输出层各节点的输出值
    for(int j = 0; j < ou_num; j++)
    {
        Type t = 0;
        for(int i = 0; i < hd_nums[k-1]; i++)
            t += w[k][i][j] * x[k-1][i];
        t += b[k][j];
        x[k][j] = t; //Activator(t); //下载的代码如此。不过输出层的激活函数会不会用线性更好? TODO
    }
}

//误差信号反向传递子过程
void BP::ReverseTransfer(int cnt)
{   
    Type tmp[NUM+1];
    Type delta[NUM+1]; 
    int k = LAYER-1, i, j;
    //delta[k] = 0.0;
    for(i = 0; i < hd_nums[k]; i++) {
        delta[i] = x[k][i] - data.at(cnt).y[i]; 
    }
    for(k = LAYER-1; k>1; k--) {

        //更新 :  w(k) = w(k) - eta * (x(k-1)*delta(k))T
        //        b(k) = b(k) - eta* (delta(k))T
        Type testEta = delta[0] * x[k-1][0]; //算算变化率是否过于夸张..
        if (testEta > 1.0) { //随便拍个数，太小的变化量可以不考虑//
            testEta = w[k][0][0] / testEta;
            if (testEta < 0)
                testEta = -testEta;
            if (testEta < TOO_SMALL) { //拍的警戒值..
                if (ETA_W > testEta) {
                    ETA_W = 0.5 * testEta; //直接testEta就改成0了，不好。这里也设个断点。考虑让ETA调成 w[k][i][j]/delta[j] * x[k-1][i]  
                    ETA_B = 0.1 * ETA_W;
                }
            }
        }
        for(i = 0; i < hd_nums[k-1]; i++)
        {
            for(j = 0; j < hd_nums[k]; j++) {
                w[k][i][j] = w[k][i][j]*(1-ETA_W*REGULAR) - ETA_W * delta[j] * x[k-1][i]; //delta[j] * x[k-1][i] 是不是反了?? 
                if(w[k][i][j] > 1.0/TOO_SMALL) 
                    ETA_W = ETA_W; //便于设断点,追查+Inf..
            }
        }
        for(i = 0; i < hd_nums[k]; i++) //按原来下载的代码中抄的是 i < hd_nums[k] ，但感觉应该k-1有效吧..
            b[k][i] -= ETA_B * delta[i];

        for(j = 0; j < hd_nums[k-1]; j++) {
            tmp[j] = 0;
            for(i = 0; i < hd_nums[k]; i++) {
                tmp[j] += delta[i]*w[k][j][i]; //不是w[k][i][j]..
            }            
            Type t = 0; //正向传播得到激活前的x[k-1]
            for(i = 0; i < hd_nums[k-2]; i++)
                t += w[k-1][i][j] * x[k-2][i];
            t += b[k-1][j];
            tmp[j] *= Diff_Activator(t);
        }
        for(j = 0; j < hd_nums[k-1]; j++) {
            delta[j] = tmp[j];
        }
    }
    //k=1;
    for(i = 0; i < hd_nums[k-1]; i++)
    {
        for(j = 0; j < hd_nums[k]; j++) {
            w[k][i][j] = w[k][i][j]*(1-ETA_W*REGULAR) - ETA_W * delta[j] * x[k-1][i]; //delta[j] * x[k-1][i] 是不是反了?? 
            if(w[k][i][j] > 1.0/TOO_SMALL) 
                ETA_W = ETA_W; //便于设断点,追查+Inf..
        }
    }
    for(i = 0; i < hd_nums[k]; i++) //按原来下载的代码中抄的是 i < hd_nums[k] ，但感觉应该k-1有效吧..
        b[k][i] -= ETA_B * delta[i];
        //delta[k] += tmp[i] * Diff_Activator(x[k][i]);
    // 第k层的delta,有0~hd_nums[k-1]共1*k维, = 第(k+1)层的delta（1*hd_nums[k+1]维） * 第k+1层的w（hd_nums[k+1}*hd_nums[k]维），后，每项乘以标量 Diff_Activator(激活前的第k层输出，即wx+b)

    /*for(k = LAYER-1; k>1; k--) { 
        for(i = 0; i < hd_nums[k]; i++) {
            tmp[i] = delta[k+1] * w[k]; //tmp是矩阵，两个维度分别是相邻两层的神经元数量..
            Type delta = tmp * Diff_Activator(x[k][i]); tmp = delta * w //各向量矩阵的大小需要确认下 TODO
    }*/
} // tmp=(y-x[k]); delta = tmp multiply Diff_Activator(x[k]); tmp = delta * w
        //CalcDelta(cnt);   UpdateNetWork();
//计算单个样本的误差
Type BP::GetError(int cnt)
{
    Type ans = 0;
    for(int i = 0; i < ou_num; i++)
        ans += 0.5 * (x[LAYER-1][i] - data.at(cnt).y[i]) * (x[LAYER-1][i] - data.at(cnt).y[i]);
    return ans;
}

//计算所有样本的精度——下载的程序这个Accu的命名不大好..但各种大小写变形用了很多就先不改了.叫Loss或Cost更好。多分类时一般建议改用交叉熵..
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
        ans += GetError(i);
        //int n = data.at(i).y.size();
        //for(int j = 0; j < n; j++) //    ans += 0.5 * (x[LAYER-1][j] - data.at(i).y[j]) * (x[LAYER-1][j] - data.at(i).y[j]);
    }
    return ans / num;
}

//计算调整量——不知对不对，已废弃重写..
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

//根据计算出的调整量对BP网络进行调整——不知对不对，已废弃重写..
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

Type BP::Sigmoid(const Type x)
{
    Type res =  A / (1 + exp(-x / B));
    return res;
}
//计算Activator函数的值
Type BP::Activator(const Type x)
{
    //Type res =  A / (1 + exp(-x / B));
    //return res;
    if (x >= 0.0)
        return A*x;
    else 
        return B*x;
}

Type BP::Diff_Activator(const Type x) {
    if (x >= 0.0)
        return A;
    else 
        return B;
    //Type t = Sigmoid(x);
    //t = t * (1.0 - t) * BB;
    //return t;
}

int main() {
    seed = time(NULL); 
    printf("seed = %d\n", seed); 
    srand(seed); 
    BP a;
    int i;
    /*************控制这里切换注释段,为啥不#ifdef..   * /
    int n = 2;
    Vector<Type> x;
    //for(i=0; i<n; i++)
    x.push_back(0);
    x.push_back(0);
//{0.1,0.2,0.3};
    printf("x.size=%lu\n", x.size());
    Vector<Type> y;
    y.push_back(0);//0.6);
    Data d;
    d.x=x;
    d.y=y;
    Vector<Data> dataset;
    dataset.push_back(d);
    d.x[0] = 0;
    d.x[1] = 1;
    d.y[0] = 1;
    dataset.push_back(d);
    d.x[0] = 1;
    d.x[1] = 0;
    d.y[0] = 1;
    dataset.push_back(d);
    d.x[0] = 1;
    d.x[1] = 1;
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
    int nd;
    for(nd=0; nd<100; nd++) {
        d.y[0] = 0.0;
        for(i=0; i<n; i++) {
            d.x[i] = getRandNum(0, 0); //1.0 - 2.0*rand()/RAND_MAX;
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

    for(nd=100; nd<200; nd++) {
        d.y[0] = 0.0;
        for(i=0; i<n; i++) {
            d.x[i] = getRandNum(0, 0); //1.0 - 2.0*rand()/RAND_MAX;
            d.y[0] += d.x[i];
        }
        if (d.y[0] > 0)
            d.y[0] = 1;
        else
            d.y[0] = 0;
        dataset.push_back(d);
    }

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

bool BP::AdjustEta(Type last_acc, Type accu) { //用局部变量覆盖同名类变量..主要是懒得替换了..
        if(last_acc < 0)
            return false;
        if(accu > last_acc) { //误差震荡，需减少学习率 :
            if(ETA_W > MIN_ETA) {
                ETA_W *=   0.5;   //权值调整率
                ETA_B *=   0.5;    //阀值调整率
            } 
            printf("eta_w = %lf, eta_b =%lg\n", ETA_W, ETA_B); 
            return true; //如果是accu，说明不能恢复学习率..
        } 
        return false;
}
