#include <iostream>
#include "../llm/context.hpp"
using namespace std;

template<typename T>
void testType(){
  switch (one::IsType<T>::value){
  case one::DTYPE::ISFLOAT:
    cout << "this is float type...\n";
    return;
  case one::DTYPE::ISINT32_T:
    cout << "this is int32_t type...\n";
    return;
  default:
    break;
  }
}

int main(){
  testType<float>();
  return 0;
}