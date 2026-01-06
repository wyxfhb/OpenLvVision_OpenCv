/**
    Copyright (C) 2023-2025, John Medland
    All rights reserved.
    https://gitlab.com/serenial/g-augmented-reality-toolkit/-/blob/main/shared-binary/g_ar_toolkit/include/g_ar_toolkit/lv_interop/lv_edvr_managed_object.hpp
*/

#pragma once

#include "EdvrTypes.h"
#include <stdexcept>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace lvi
{
    namespace edvr
    {
        // --- Exception Class ---
        /**
         * @brief Exception thrown when an invalid EDVR reference is used.
         */
        struct InvalidReferenceException : public std::exception
        {
            const char* what() const throw()
            {
                return "Invalid EDVR-Object Reference. The object may have been deleted.";
            }
        };

        /**
         * @brief Thread-safe C++ template wrapper for EDVR-managed objects.
         * Guarantees mutually exclusive access between C++ and LabVIEW callbacks.
         */
        template <typename T>
        class EdvrObject
        {

#pragma region Private Members and Types
        private:
            /**
             * @brief Defines who currently holds the exclusive lock.
             */
            enum class LockState
            {
                NONE,    // Resource is free
                LABVIEW, // LabVIEW holds the lock
                CPP      // A C++ EdvrObject instance holds the lock
            };

            /**
             * @brief Holds the managed object (T*) and synchronization tools.
             * The std::mutex ensures only one thread can check/change 'locked' or wait on 'cv' at once.
             */
            struct PersistentData
            {
                T* object = nullptr;
                LockState locked = LockState::NONE;
                std::mutex mutex; // The gatekeeper for accessing 'locked' and using 'cv'
                std::condition_variable cv; // Used by threads to sleep until notified of a state change

                PersistentData(T* obj) : object(obj) {}
                ~PersistentData()
                {
                    delete object;
                    object = nullptr;
                }
            };

            // --- Private Member Variables ---
            ReferencePtr const edvrRefPtr; // Pointer to the LV Refnum itself
            Context const ctx;             // LV Execution Context (0 for Create, valid for Access)
            DataPtr const edvrDataPtr;     // Pointer to LV's internal EDVR data struct
            PersistentData* const persistentData; // Pointer to our C++ struct containing T* and mutex

#pragma endregion Private Members and Types


#pragma region Callbacks for LabVIEW
        private:
            /**
             * @brief LabVIEW Lock Callback. Ensures LabVIEW waits if C++ has the lock.
             */
            static MgErr _FUNCC onLabviewLock(DataPtr ptr)
            {
                PersistentData* pd = reinterpret_cast<PersistentData*>(ptr->metadata_ptr);
                if (!pd) { return bogusError; }

                // Acquire the mutex. Only one thread (LV or C++) can hold this at a time.
                std::unique_lock<std::mutex> lock(pd->mutex);

                // While waiting, the mutex is automatically released, allowing C++ to unlock.
                pd->cv.wait(lock, [&] { return pd->locked != LockState::CPP; });
                // When wait returns, we hold the mutex again, and know C++ is not locking.

                // Claim the lock for LabVIEW if it's currently free.
                if (pd->locked == LockState::NONE)
                {
                    pd->locked = LockState::LABVIEW;
                }
                // Mutex is automatically released when 'lock' goes out of scope.
                return mgNoErr;
            }

            /**
             * @brief LabVIEW Unlock Callback. Releases LV lock and notifies C++.
             */
            static MgErr _FUNCC onLabviewUnlock(DataPtr ptr)
            {
                PersistentData* pd = reinterpret_cast<PersistentData*>(ptr->metadata_ptr);
                if (!pd) { return bogusError; }

                bool notify = false; // Flag to notify outside the lock scope
                {
                    // Acquire mutex to safely change state
                    std::lock_guard<std::mutex> lock(pd->mutex);
                    // Only change state if LabVIEW actually held the lock
                    if (pd->locked == LockState::LABVIEW)
                    {
                        pd->locked = LockState::NONE; // Release the lock
                        notify = true; // Signal that we need to notify waiters
                    }
                } // Mutex released here

                // If we changed the state, wake up any threads (C++ or Delete)
                // that might be waiting for the lock to become free.
                if (notify) {
                    pd->cv.notify_all();
                }
                return mgNoErr;
            }

            /**
             * @brief LabVIEW Delete Callback. Ensures deletion only happens when C++ is done.
             */
            static void _FUNCC onLabviewDelete(DataPtr ptr)
            {
                PersistentData* pd = reinterpret_cast<PersistentData*>(ptr->metadata_ptr);
                if (!pd) { return; }

                // Acquire mutex.
                std::unique_lock<std::mutex> lock(pd->mutex);

                // Wait efficiently IF AND ONLY IF C++ currently holds the lock.
                // Mutex is released while waiting.
                pd->cv.wait(lock, [&] { return pd->locked != LockState::CPP; });
                // When wait returns, we hold the mutex, and know C++ is not locking.

                // Now it's safe to delete.
                lock.unlock(); // IMPORTANT: Release the mutex *before* deleting 'pd'.
                delete pd;     // This deletes the PersistentData, which deletes the T object.
                ptr->metadata_ptr = reinterpret_cast<uintptr_t>(nullptr); // Clear pointer in LV struct
            }
#pragma endregion C-style Callbacks for LabVIEW

#pragma region Private Helper Functions
        private:
            /**
             * @brief C++ Acquire Lock Helper. Ensures C++ waits if LabVIEW (or another C++ instance) has the lock.
             */
            static void lockCppSide(PersistentData* data)
            {
                if (!data) { throw InvalidReferenceException(); }

                // Acquire mutex.
                std::unique_lock<std::mutex> lock(data->mutex);

                // Wait efficiently IF AND ONLY IF the lock is NOT free (held by LV or another C++).
                // Mutex is released while waiting.
                data->cv.wait(lock, [&] { return data->locked == LockState::NONE; });
                // When wait returns, we hold the mutex, and know the state was NONE.

                // Claim the lock for C++.
                data->locked = LockState::CPP;
                // The unique_lock 'lock' continues to hold the mutex until the
                // EdvrObject instance goes out of scope (destructor runs).
            }

            /**
             * @brief C++ Release Lock Helper. Releases C++ lock and notifies LabVIEW.
             */
            static void unlockCppSide(PersistentData* data)
            {
                if (!data) { return; } // Object might already be deleted

                bool notify = false; // Flag to notify outside lock scope
                {
                    // Acquire mutex to safely change state
                    std::lock_guard<std::mutex> lock(data->mutex);

                    // Only change state if C++ actually held the lock
                    if (data->locked == LockState::CPP)
                    {
                        data->locked = LockState::NONE; // Release the lock
                        notify = true; // Signal that we need to notify waiters
                    }
                } // Mutex released here

                // If we changed the state, wake up any threads (LabVIEW Lock/Delete)
                // that might be waiting for the lock to become free.
                if (notify) {
                    data->cv.notify_all();
                }
            }


            // --- Other Private Helper Functions (Error Checking) ---
            Context getCtx()
            {
                Context context;
                if (EDVR_GetCurrentContext(&context) != mgNoErr)
                {
                    throw std::runtime_error("Failed to get EDVR context.");
                }
                return context;
            }

            DataPtr createNewEdvrDataPtr(T* object)
            {
                DataPtr dataPtr = nullptr;
                if (EDVR_CreateReference(this->edvrRefPtr, &dataPtr) != mgNoErr)
                {
                    delete object; // Clean up T if LV call fails
                    throw std::runtime_error("Unable to create EDVR reference.");
                }
                // Ensure dataPtr is not null after successful creation
                if (!dataPtr)
                {
                    delete object; // Clean up T if LV call fails unexpectedly
                    throw std::runtime_error("EDVR_CreateReference succeeded but returned null pointer.");
                }

                PersistentData* pd = nullptr;
                try {
                    pd = new PersistentData(object); // Create our management struct
                }
                catch (...) {
                    // If 'new PersistentData' fails, try to clean up the LV ref (best effort)
                    EDVR_ReleaseRefWithContext(*this->edvrRefPtr, 0);
                    delete object; // Clean up T
                    throw; // Re-throw the exception (likely bad_alloc)
                }

                dataPtr->metadata_ptr = reinterpret_cast<uintptr_t>(pd);
                dataPtr->lock_callback_fn_ptr = &EdvrObject::onLabviewLock;
                dataPtr->unlock_callback_fn_ptr = &EdvrObject::onLabviewUnlock;
                dataPtr->delete_callback_fn_ptr = &EdvrObject::onLabviewDelete;
                return dataPtr;
            }

            DataPtr getEdvrDataPtr()
            {
                DataPtr dataPtr = nullptr;
                MgErr err = EDVR_AddRefWithContext(*this->edvrRefPtr, this->ctx, &dataPtr);

                // Primary check: Did LabVIEW report the reference as invalid?
                if (err != mgNoErr)
                {
                    throw InvalidReferenceException();
                }

                // Secondary checks: Ensure pointer and metadata are valid
                if (dataPtr == nullptr || dataPtr->metadata_ptr == 0)
                {
                    // Release the ref we just added if something is wrong
                    EDVR_ReleaseRefWithContext(*this->edvrRefPtr, this->ctx);
                    throw InvalidReferenceException();
                }
                return dataPtr;
            }
#pragma endregion Private Helper Functions

#pragma region Public Interface (Constructors, Destructor, Operators)
        public:
            /**
             * @brief **Create Constructor**
             * Creates the C++ object & LV reference, claims initial C++ lock.
             */
            EdvrObject(ReferencePtr lvRefFromLabview, T* object) :
                edvrRefPtr(lvRefFromLabview),
                ctx(0), // No context needed for creation
                edvrDataPtr(createNewEdvrDataPtr(object)), // Creates LV Ref & PersistentData
                persistentData(reinterpret_cast<PersistentData*>(edvrDataPtr->metadata_ptr))
            {
                // Safety check, though createNewEdvrDataPtr should throw if pd fails
                if (!persistentData)
                {
                    throw std::runtime_error("Failed to initialize persistent data during creation.");
                }
                // Object is newly created, C++ claims the initial lock.
                persistentData->locked = LockState::CPP;
            }

            /**
             * @brief **Access Constructor**
             * Gets existing LV reference, acquires C++ lock (waits if necessary).
             */
            EdvrObject(ReferencePtr lvRefFromLabview) :
                edvrRefPtr(lvRefFromLabview),
                ctx(getCtx()), // Get LV context
                edvrDataPtr(getEdvrDataPtr()), // Gets LV Ref, checks validity
                persistentData(reinterpret_cast<PersistentData*>(edvrDataPtr->metadata_ptr))
            {
                // Acquire the C++ lock for the duration of this object's lifetime.
                // This function handles waiting if LabVIEW currently holds the lock.
                lockCppSide(persistentData);
            }

            /**
             * @brief **Destructor**
             * Releases the C++ lock and notifies waiters. Releases LV reference if added by Access constructor.
             */
            ~EdvrObject()
            {
                // Release the C++ lock and notify any waiting LabVIEW threads
                unlockCppSide(persistentData);

                // If this object was created via the Access constructor (ctx != 0),
                // release the reference count we added in getEdvrDataPtr.
                if (ctx != 0)
                {
                    EDVR_ReleaseRefWithContext(*edvrRefPtr, ctx);
                }
                // Note: The actual T* object is deleted ONLY by onLabviewDelete when LV destroys the DVR.
            }

            // --- C++ Object Access Operators ---
            // These are safe to call because the Access Constructor acquired the lock,
            // and the Destructor will release it.
            T* operator->()
            {
                if (!persistentData) throw InvalidReferenceException();
                return persistentData->object;
            }

            const T* operator->() const
            {
                if (!persistentData) throw InvalidReferenceException();
                return persistentData->object;
            }

            T& operator*()
            {
                if (!persistentData) throw InvalidReferenceException();
                return *(persistentData->object);
            }

            const T& operator*() const
            {
                if (!persistentData) throw InvalidReferenceException();
                return *(persistentData->object);
            }
#pragma endregion Public Interface (Constructors, Destructor, Operators)

        }; // End Class EdvrObject

    } // namespace edvr
} // namespace lvi