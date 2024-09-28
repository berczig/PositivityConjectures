#ifndef CACHE_DECORATOR_H
#define CACHE_DECORATOR_H

#include <functional>
#include <map>
#include <tuple>

using namespace std;

template <typename R, typename... A>
class CacheDecorator {
  public:
    CacheDecorator(){}
    // Constructor accepting a function to decorate
    CacheDecorator(function<R(A...)> f) : f_(f) {}

    // Operator overloading to call the decorated function with caching
    R operator()(A... a) {
        tuple<A...> key(a...);
        auto search = map_.find(key);
        if (search != map_.end()) {
            return search->second;
        }

        auto result = f_(a...);
        map_[key] = result;
        return result;
    }

  private:
    function<R(A...)> f_;
    map<tuple<A...>, R> map_;
};

#endif // CACHE_DECORATOR_H


