#include <iostream>
#include <vector>
#include <string>
#include <memory> // For std::unique_pt
#include <map>
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
        A(){

        }
        unique_ptr<B> b;

        void create(B& in) {
            b = make_unique<B>(in);
        };

        void precreate(){
            B b = B(42);
            create(b);
        }
        void use(){
            cout << b->val << endl;
        };
};


// int main(){
//     shared_ptr<B> b = make_shared<B>(29);
//     shared_ptr<B> b2 = b;
// }





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


// int main(){
//     A a = A();
//     a.create();
//     a.use();
// }