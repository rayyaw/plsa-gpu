#pragma once

#include <string.h>

namespace utils {
template <typename T>

// Simple struct for returning a list's size along with the list itself
// Managing the number of items is left to the user, and all functions implicitly assume it is correct
// We also assume that if num_items is 0, items is nullptr
class ListWithSize {
    public:
    size_t num_items = 0;
    T* items = nullptr;

    ListWithSize<T>() = default;

    ListWithSize<T>(size_t nitems) {
        num_items = nitems;
        items = new T[num_items];
    };

    ListWithSize<T>(size_t nitems, T foreign_items[]) {
        num_items = nitems;
        items = foreign_items;
        manual_alloc = false;
    }

    ListWithSize<T>(const ListWithSize<T> &other) {
        num_items = other.num_items;
        items = new T[num_items];
        memcpy(items, other.items, sizeof(T) * num_items);
    }

    ~ListWithSize<T>() {
        if (manual_alloc) delete[] items;
    }

    void setLst(T* foreign_items) {
        items = foreign_items;
        manual_alloc = false;
    }

    private:
    bool manual_alloc = true;
};

template <typename T>
ListWithSize<T> makeEmptyList() {
    ListWithSize<T> list = ListWithSize<T>();
    list.num_items = 0;
    list.items = (T*) nullptr;
    return list;
}
}