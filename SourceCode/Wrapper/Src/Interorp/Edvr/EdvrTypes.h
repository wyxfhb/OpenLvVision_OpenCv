/**
    Copyright (C) 2023-2025, John Medland
    All rights reserved.
    https://gitlab.com/serenial/g-augmented-reality-toolkit/-/blob/main/shared-binary/g_ar_toolkit/include/g_ar_toolkit/lv_interop/lv_types.hpp
*/

#pragma once
#pragma comment(lib, "labview.lib")
#include "extcode.h" 

namespace lvi
{
    namespace edvr
    {
        // --- EDVR Type Definitions ---

        // Forward declaration
        struct Data;

        // Type aliases
        typedef Data* DataPtr;
        typedef MgErr(_FUNCC* LockCallbackPtr)(DataPtr);
        typedef void (_FUNCC* DeleteCallbackPtr)(DataPtr);

        // C++ definition of the LabVIEW EDVR Data struct
        struct Data
        {
            uintptr_t metadata_ptr;
            int32 n_dims;
            struct LV_EDVRNDimensionalSubArray_t {
                uintptr_t* data_ptr;
                struct LV_EDVRDimensionSpecifier_t {
                    size_t dimension_size;
                    ptrdiff_t stride;
                } dimension_specifier[5];
            } sub_array;
            DeleteCallbackPtr delete_callback_fn_ptr;
            LockCallbackPtr lock_callback_fn_ptr;
            LockCallbackPtr unlock_callback_fn_ptr;
        };

        // Type aliases for MagicCookie types
        typedef MagicCookie Context;
        typedef MagicCookie Reference;
        typedef MagicCookie* ReferencePtr;

    } // namespace edvr
} // namespace lvi


extern "C" {   
    MgErr _FUNCC EDVR_GetCurrentContext(lvi::edvr::Context* context);
    MgErr _FUNCC EDVR_CreateReference(lvi::edvr::ReferencePtr ref, lvi::edvr::DataPtr* data_ptr);
    MgErr _FUNCC EDVR_AddRefWithContext(lvi::edvr::Reference ref, lvi::edvr::Context context, lvi::edvr::DataPtr* data_ptr);
    MgErr _FUNCC EDVR_ReleaseRefWithContext(lvi::edvr::Reference ref, lvi::edvr::Context context);
}