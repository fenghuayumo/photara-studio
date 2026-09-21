#include "ba/bearing_cuda.hpp"
#include "ba/optimizer.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <fstream>
#include <ceres/ceres.h>

struct BearingCost {
    std::array<double,3> d;
    template<class T> bool operator()(const T* c,const T* p,T* r)const{
        T x[3],length=T(0);for(int k=0;k<3;++k){x[k]=p[k]-c[k];length+=x[k]*x[k];}
        length=ceres::sqrt(length);for(int k=0;k<3;++k)r[k]=T(d[k])-x[k]/length;return true;
    }
};
struct BaselineCost {
    double length;
    template<class T>bool operator()(const T* a,const T* b,T* r)const{
        T d=T(0);for(int k=0;k<3;++k)d+=(a[k]-b[k])*(a[k]-b[k]);r[0]=ceres::sqrt(d)-T(length);return true;
    }
};

ceres::Solver::Summary solve_reference(photara::ba::BearingProblem& p,
    const std::vector<double>& weights,int iterations,bool scaling=true){
    ceres::Problem reference;
    for(unsigned i=0;i<p.observations.size();++i){const auto& o=p.observations[i];
        reference.AddResidualBlock(new ceres::AutoDiffCostFunction<BearingCost,3,3,3>(new BearingCost{o.direction}),
            new ceres::ScaledLoss(new ceres::HuberLoss(p.huber),weights[i],ceres::TAKE_OWNERSHIP),
            p.cameras[o.camera].data(),p.points[o.point].data());}
    reference.SetParameterBlockConstant(p.cameras[p.anchor].data());
    reference.AddResidualBlock(new ceres::AutoDiffCostFunction<BaselineCost,1,3,3>(new BaselineCost{p.baseline}),
        nullptr,p.cameras[p.baseline_first].data(),p.cameras[p.baseline_second].data());
    ceres::Solver::Options opt;opt.linear_solver_type=ceres::SPARSE_SCHUR;opt.max_num_iterations=iterations;
    opt.jacobi_scaling=scaling;
    opt.function_tolerance=1e-10;ceres::Solver::Summary summary;ceres::Solve(opt,&reference,&summary);
    return summary;
}

int main(int argc,char** argv){try{
    using namespace photara::ba;
    if(!CudaOptimizer::is_available())return 0;
    if(argc==2){
        std::ifstream f(argv[1],std::ios::binary);std::uint64_t sizes[3]{};std::uint32_t ids[3]{};double values[2]{};
        f.read(reinterpret_cast<char*>(sizes),sizeof(sizes));f.read(reinterpret_cast<char*>(ids),sizeof(ids));
        f.read(reinterpret_cast<char*>(values),sizeof(values));
        if(!f||sizes[0]>2000||sizes[1]>1000000||sizes[2]>10000000)throw std::runtime_error("Bad fixture");
        BearingProblem original;original.anchor=ids[0];original.baseline_first=ids[1];original.baseline_second=ids[2];
        original.baseline=values[0];original.huber=values[1];original.cameras.resize(sizes[0]);original.points.resize(sizes[1]);original.observations.resize(sizes[2]);
        f.read(reinterpret_cast<char*>(original.cameras.data()),sizes[0]*sizeof(original.cameras[0]));
        f.read(reinterpret_cast<char*>(original.points.data()),sizes[1]*sizeof(original.points[0]));
        f.read(reinterpret_cast<char*>(original.observations.data()),sizes[2]*sizeof(original.observations[0]));
        if(!f)throw std::runtime_error("Truncated fixture");
        std::vector<double> weights(sizes[2],1);
        for(int count:{1,8,30}){
            auto cpu=original,gpu=original;const auto c=solve_reference(cpu,weights,count);
            CudaBearingOptimizer solver(gpu);const auto g=solver.solve(weights,count,1e-10,0);solver.download(gpu);
            double error=0;unsigned worst=0;
            for(unsigned i=0;i<cpu.cameras.size();++i)for(int k=0;k<3;++k)if(std::abs(cpu.cameras[i][k]-gpu.cameras[i][k])>error){error=std::abs(cpu.cameras[i][k]-gpu.cameras[i][k]);worst=i;}
            std::cout<<"fixture iterations="<<count<<" CPU="<<c.final_cost<<" GPU="<<g.final_cost<<" camera_max_difference="<<error<<" worst="<<worst<<std::endl;
            if(!g.usable||!c.IsSolutionUsable()||error>1e-6||std::abs(c.final_cost-g.final_cost)>1e-8)return 6;
        }
        return 0;
    }
    BearingProblem p;p.anchor=3;p.baseline_first=1;p.baseline_second=7;p.huber=.03;
    for(int c=0;c<12;++c)p.cameras.push_back({2*std::cos(c*.5),2*std::sin(c*.5),.2*std::sin(c*.9)});
    std::mt19937 rng(19);std::uniform_real_distribution<double> random(-1,1);
    for(int i=0;i<300;++i)p.points.push_back({random(rng),random(rng),4+random(rng)});
    double baseline2=0;for(int k=0;k<3;++k){double d=p.cameras[1][k]-p.cameras[7][k];baseline2+=d*d;}p.baseline=std::sqrt(baseline2);
    for(unsigned i=0;i<p.points.size();++i)for(unsigned c=0;c<p.cameras.size();++c){
        std::array<double,3> d;double n=0;for(int k=0;k<3;++k){d[k]=p.points[i][k]-p.cameras[c][k];n+=d[k]*d[k];}
        for(auto& v:d)v/=std::sqrt(n);p.observations.push_back({c,i,d});}
    const auto truth=p;
    for(unsigned c=0;c<p.cameras.size();++c)if(c!=p.anchor)for(auto& v:p.cameras[c])v+=.08*random(rng);
    for(auto& x:p.points)for(auto& v:x)v+=.08*random(rng);
    const auto initial=p;
    CudaBearingOptimizer solver(p);std::vector<double> weights(p.observations.size(),1);
    const auto s=solver.solve(weights,80,1e-10,0);solver.download(p);
    double maximum=0;for(unsigned c=0;c<p.cameras.size();++c)for(int k=0;k<3;++k)
        maximum=std::max(maximum,std::abs(p.cameras[c][k]-truth.cameras[c][k]));
    std::cout<<"cost="<<s.initial_cost<<" -> "<<s.final_cost<<" camera_max_error="<<maximum<<" iterations="<<s.iterations<<" seconds="<<s.seconds<<'\n';
    if(!s.usable||s.final_cost>1e-12||maximum>1e-5||p.cameras[p.anchor]!=truth.cameras[p.anchor])return 2;
    weights[0]=-1;try{solver.solve(weights,1,1e-8,0);return 3;}catch(const std::invalid_argument&){}
    weights[0]=1;
    auto step_cpu=initial,step_gpu=initial;
    solve_reference(step_cpu,weights,1);
    CudaBearingOptimizer one_step(step_gpu);one_step.solve(weights,1,1e-10,0);one_step.download(step_gpu);
    double step_error=0;
    for(unsigned c=0;c<step_cpu.cameras.size();++c)for(int k=0;k<3;++k)
        step_error=std::max(step_error,std::abs(step_cpu.cameras[c][k]-step_gpu.cameras[c][k]));
    for(unsigned i=0;i<step_cpu.points.size();++i)for(int k=0;k<3;++k)
        step_error=std::max(step_error,std::abs(step_cpu.points[i][k]-step_gpu.points[i][k]));
    std::cout<<"one-step CPU/GPU parameter max difference="<<step_error<<'\n';
    if(step_error>1e-8)return 5;
    // Force small Jacobian columns to exercise the scaled LM diagonal floor.
    // Well-conditioned unit-scale geometry does not expose this discrepancy.
    auto scaled_cpu=initial;
    for(auto& c:scaled_cpu.cameras)for(auto& v:c)v*=1000;
    for(auto& x:scaled_cpu.points)for(auto& v:x)v*=1000;
    scaled_cpu.baseline*=1000;auto scaled_gpu=scaled_cpu;
    const auto scaled_reference=solve_reference(scaled_cpu,weights,8);
    CudaBearingOptimizer scaled_solver(scaled_gpu);
    const auto scaled_result=scaled_solver.solve(weights,8,1e-10,0);scaled_solver.download(scaled_gpu);
    double scaled_error=0;
    for(unsigned c=0;c<scaled_cpu.cameras.size();++c)for(int k=0;k<3;++k)
        scaled_error=std::max(scaled_error,std::abs(scaled_cpu.cameras[c][k]-scaled_gpu.cameras[c][k]));
    for(unsigned i=0;i<scaled_cpu.points.size();++i)for(int k=0;k<3;++k)
        scaled_error=std::max(scaled_error,std::abs(scaled_cpu.points[i][k]-scaled_gpu.points[i][k]));
    std::cout<<"scaled weak-direction CPU/GPU parameter max difference="<<scaled_error<<'\n';
    if(!scaled_reference.IsSolutionUsable()||!scaled_result.usable||scaled_error>1e-6)return 7;
    for(unsigned i=0;i<p.observations.size();++i){
        auto& d=p.observations[i].direction;double norm=0;
        for(auto& v:d){v+=(i%17==0?.15:.001)*random(rng);norm+=v*v;}
        for(auto& v:d)v/=std::sqrt(norm);weights[i]=.2+.8*(i%7)/6.;
    }
    auto cpu=p;ceres::Problem reference;
    for(unsigned i=0;i<cpu.observations.size();++i){const auto& o=cpu.observations[i];
        reference.AddResidualBlock(new ceres::AutoDiffCostFunction<BearingCost,3,3,3>(new BearingCost{o.direction}),
            new ceres::ScaledLoss(new ceres::HuberLoss(cpu.huber),weights[i],ceres::TAKE_OWNERSHIP),
            cpu.cameras[o.camera].data(),cpu.points[o.point].data());}
    reference.SetParameterBlockConstant(cpu.cameras[cpu.anchor].data());
    reference.AddResidualBlock(new ceres::AutoDiffCostFunction<BaselineCost,1,3,3>(new BaselineCost{cpu.baseline}),
        nullptr,cpu.cameras[cpu.baseline_first].data(),cpu.cameras[cpu.baseline_second].data());
    ceres::Solver::Options opt;opt.linear_solver_type=ceres::SPARSE_SCHUR;opt.max_num_iterations=80;
    opt.function_tolerance=1e-10;ceres::Solver::Summary summary;ceres::Solve(opt,&reference,&summary);
    CudaBearingOptimizer robust(p);const auto robust_result=robust.solve(weights,80,1e-10,0);robust.download(p);
    double difference=std::abs(robust_result.final_cost-summary.final_cost);
    std::cout<<"weighted Huber CPU="<<summary.final_cost<<" GPU="<<robust_result.final_cost<<" difference="<<difference<<'\n';
    if(!robust_result.usable||!summary.IsSolutionUsable()||difference>1e-7||p.cameras[p.anchor]!=truth.cameras[p.anchor])return 4;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
