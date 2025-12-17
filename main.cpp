#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <CL/cl.h>

enum M{CPU,GPU,HELP};
struct O{M m=CPU;double s=1.0;};

void help(){
 std::cout <<
 "Usage:\n"
 "  --cpu        CPU benchmark\n"
 "  --gpu        GPU(OpenCL) benchmark\n"
 "  --s <sec>    duration\n"
 "  --help       show this\n";
 exit(0);
}

O arg(int c,char**v){
 O o;
 for(int i=1;i<c;i++){
  std::string a=v[i];
  if(a=="--cpu")o.m=CPU;
  else if(a=="--gpu")o.m=GPU;
  else if(a=="--help")o.m=HELP;
  else if(a=="--s"&&i+1<c)o.s=atof(v[++i]);
  else{std::cerr<<"bad arg\n";exit(1);}
 }
 return o;
}

/* CPU */
void run_cpu(double s){
 const size_t N=1e6;
 std::vector<double>a(N),b(N),c(N);
 for(size_t i=0;i<N;i++)a[i]=b[i]=i;
 size_t cnt=0;
 auto st=std::chrono::high_resolution_clock::now();
 for(;;){
  for(size_t i=0;i<N;i++)c[i]=a[i]+b[i];
  cnt+=N;
  double t=std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now()-st).count();
  if(t>=s){
   double f=cnt/t;
   std::cout<<"CPU\n"<<f/1e9<<" GFLOPS ("<<f/1e12<<" TFLOPS)\n";
   break;
  }
 }
}

/* GPU(OpenCL) */
const char*src=R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void k(__global const double*a,__global const double*b,__global double*c){
 int i=get_global_id(0);c[i]=a[i]+b[i];
})";

void run_gpu(double s){
 cl_platform_id p;cl_uint pc;
 if(clGetPlatformIDs(1,&p,&pc)!=CL_SUCCESS||pc==0){
  std::cerr<<"No OpenCL platform\n";return;
 }

 cl_device_id d;
 cl_int r=clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,nullptr);
 if(r!=CL_SUCCESS){
  std::cerr<<"GPU not found, fallback CPU OpenCL\n";
  r=clGetDeviceIDs(p,CL_DEVICE_TYPE_CPU,1,&d,nullptr);
  if(r!=CL_SUCCESS){std::cerr<<"No OpenCL device\n";return;}
 }

 cl_context ctx=clCreateContext(nullptr,1,&d,nullptr,nullptr,&r);
 if(r!=CL_SUCCESS){std::cerr<<"Context fail\n";return;}

 cl_command_queue q=clCreateCommandQueue(ctx,d,0,&r);

 const size_t N=1e6,B=N*sizeof(double);
 cl_mem a=clCreateBuffer(ctx,CL_MEM_READ_ONLY,B,nullptr,&r);
 cl_mem b=clCreateBuffer(ctx,CL_MEM_READ_ONLY,B,nullptr,&r);
 cl_mem c=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,B,nullptr,&r);

 std::vector<double>h(N);for(size_t i=0;i<N;i++)h[i]=i;
 clEnqueueWriteBuffer(q,a,1,0,B,h.data(),0,nullptr,nullptr);
 clEnqueueWriteBuffer(q,b,1,0,B,h.data(),0,nullptr,nullptr);

 cl_program pr=clCreateProgramWithSource(ctx,1,&src,nullptr,&r);
 clBuildProgram(pr,1,&d,nullptr,nullptr,nullptr);
 cl_kernel k=clCreateKernel(pr,"k",&r);
 clSetKernelArg(k,0,sizeof(a),&a);
 clSetKernelArg(k,1,sizeof(b),&b);
 clSetKernelArg(k,2,sizeof(c),&c);

 size_t g=N,cnt=0;
 auto st=std::chrono::high_resolution_clock::now();
 for(;;){
  clEnqueueNDRangeKernel(q,k,1,nullptr,&g,nullptr,0,nullptr,nullptr);
  clFinish(q);cnt+=N;
  double t=std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now()-st).count();
  if(t>=s){
   double f=cnt/t;
   std::cout<<"OpenCL\n"<<f/1e9<<" GFLOPS ("<<f/1e12<<" TFLOPS)\n";
   break;
  }
 }

 clReleaseKernel(k);clReleaseProgram(pr);
 clReleaseMemObject(a);clReleaseMemObject(b);clReleaseMemObject(c);
 clReleaseCommandQueue(q);clReleaseContext(ctx);
}

int main(int c,char**v){
 auto o=arg(c,v);
 if(o.m==HELP)help();
 o.m==CPU?run_cpu(o.s):run_gpu(o.s);
}
