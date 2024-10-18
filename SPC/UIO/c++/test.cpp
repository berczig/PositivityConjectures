#include <iostream>
#include <vector>
#include <string>
#include <memory> // For std::unique_pt
#include <map>
#include <windows.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <tuple>
#include "CoreGenerator.h"
#include <unordered_map>
using namespace std;

// class A{
//     public:
//         int val;
//         A(int val){
//             this->val = val;
//             cout << "construct A with " << val << endl;
//         }

//         A(const A& other){
//             cout << "copy a\n";
//             val = other.val;
//         }
//         // A(){
//         //     cout << "construct A" << endl;
//         // }
// };

// class B{
//     public:
//         A a;
//         B(A in) : a(in){
//             a = in;
//             cout << "construct B with " << a.val << endl;
//             a.val = -77;
//         }
// };

// int main(){
//     A a = A(13);;
//     B b = B(a);
//     cout << "hey " << a.val << endl;
// }


class C{
    public:
        int val;
        C(){}
        C(int in){
            val = in;
            cout << "C-create" << endl;
        }
        C(const C& other){
            val = other.val;
            cout << "C-copy" << endl;
        }
        // Deconstructor
        ~ C() {
            cout << "C-death" << endl;
        }
};
class B{
    public:
    int val;
        B(){}
        B(int x){
            val = x;
            cout  << "B-create " << val  << endl;
        }

        B(const B& other) : val(other.val){
            cout << "B-copy from " << other.val << endl;
        }

        ~ B() {
            cout << "B-death " << this << " " << val << endl;
        }
      
};
class A{
    public:
        B b;
        A() : b(10){}

        A(const A& other){
            cout << "A-copy from " << &other << endl;
        }


};


vector<A> func(){
    vector<A> vec;
    vec.reserve(10);
    vec.emplace_back();
    return vec;
}

// int main(){
//     auto start = std::chrono::system_clock::now();

//     map<coreRepresentation, int> m;

//     vector<unsigned char> a__;
//     a__.push_back(12);
//     coreRepresentation a = coreRepresentation(a__);

//     vector<unsigned char> b__;
//     b__.push_back(12);
//     coreRepresentation b = coreRepresentation(b__);

//     m[a] = 10;


//     boolean boo = (m.find(b) == m.end());

//     cout << (boo ? "1" : "0") << endl;

//     auto end = std::chrono::system_clock::now();
//     std::chrono::duration<double> elapsed_seconds = end-start;
//     cout << "Elapsed Time: " << elapsed_seconds.count() << " seconds" << endl;
// }

// shared_ptr<B> b = make_shared<B>(100);
    // A a = A(b);
    // A a2 = a;
    // a.print();
    // a2.print();
    // cout << b.use_count() << endl;





// class C{
//     public:
//         int val;
//         C(int in){
//             val = in;
//             cout << "C-create" << endl;
//         }
//         C(const C& other){
//             val = other.val;
//             cout << "C-copy" << endl;
//         }
//         ~ C() {
//             cout << "C-death" << endl;
//         }
// };

// class B{
//     public:
//         C c;
//         B(C& c_in) : c(c_in) {}
      
// };

// class A{
//     public:
//         A(){}
//         vector<B> bs;
//         void create(){
//             C c = C(10);
//             B b = B(c);
//             cout << "here\n";
//             bs.push_back(b);
//             cout << "hxxx\n";
//         };
//         void use(){
//             for (B& b : bs){
//                 cout << "val: " << b.c.val << endl;
//             }
//         };

// };


int* funca(){
    int a[10];
    return a;
}


// int main(){
//     int* a = funca();
//     cout << a << " " << *a << " " << &a << endl;
// }