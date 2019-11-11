
/*
Install boost compute:
    https://www.boost.org/doc/libs/1_65_1/libs/compute/doc/html/boost_compute/getting_started.html

Build:
    g++  cpugpumemorycopy.cpp  -o cpugpu  -I ../../boost/boost_1_71_0/boost -lOpenCL

Data on i7-7700K and GTX1060
4096X4096:
CPU2CPU copy: 0.00467676s
CPU2GPU copy: 0.00698211s
GPU2GPU copy: 0.000133137s
GPU2CPU copy: 0.00740875s

1024X1024:
CPU2CPU copy: 0.000394094s
CPU2GPU copy: 0.000688026s
GPU2GPU copy: 9.2974e-05s
GPU2CPU copy: 0.000515176s

Code is from:
    https://stackoverflow.com/questions/47648900/memory-copy-speed-comparison-cpu-gpu
*/

#include <algorithm>
#include <boost/compute.hpp>
#include <chrono>
#include <iostream>
#include <vector>
namespace compute = boost::compute;
using namespace std::chrono;
using namespace std;

void envInformation() {

    /// // get the default compute device
    compute::device device = boost::compute::system::default_device();
    ///
    /// // print the name of the device
    std::cout << "default device: " << device.name() << std::endl;
    std::cout << "device count: " << boost::compute::system::device_count << std::endl;
}


int main() {
  int sz = 1 * 1024 * 1024;
  envInformation();
  std::vector<float> v1(sz, 2.3f), v2(sz);
  compute::vector<float> v3(sz), v4(sz);
  {
    auto s = system_clock::now();
    std::copy(v1.begin(), v1.end(), v2.begin());
    auto e = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = e - s;
    std::cout << "CPU2CPU copy: " << elapsed_seconds.count() << "s\n";
  }
  {
    auto s = system_clock::now();
    compute::copy(v1.begin(), v1.end(), v3.begin());
    auto e = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = e - s;
    std::cout << "CPU2GPU copy: " << elapsed_seconds.count() << "s\n";
  }
  {
    auto s = system_clock::now();
    compute::copy(v3.begin(), v3.end(), v4.begin());
    auto e = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = e - s;
    std::cout << "GPU2GPU copy: " << elapsed_seconds.count() << "s\n";
  }
  {
    auto s = system_clock::now();
    compute::copy(v3.begin(), v3.end(), v1.begin());
    auto e = system_clock::now();
    std::chrono::duration<double> elapsed_seconds = e - s;
    std::cout << "GPU2CPU copy: " << elapsed_seconds.count() << "s\n";
  }
  return 0;
}
