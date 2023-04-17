#pragma once

namespace utils {
template <typename T>

// Simple struct for returning a list's size along with the list itself
// Managing the number of items is left to the user, and all functions implicitly assume it is correct
struct ListWithSize {
    size_t num_items;
    T* items;
};

template <typename T>
ListWithSize<T> makeEmptyList() {
    ListWithSize<T> list = ListWithSize<T>();
    list.num_items = 0;
    list.items = (T*) nullptr;
    return list;
}
}