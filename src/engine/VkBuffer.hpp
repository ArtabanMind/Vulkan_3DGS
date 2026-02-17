// ============================================================
// File: src/engine/VkBuffer.hpp
// Role: GPU buffer creation utilities (SSBO, staging)
// ============================================================
#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>  // memcpy

namespace gs {

// ------------------------------------------------------------
// findMemoryType: GPU 메모리 타입 찾기
// ------------------------------------------------------------
// GPU는 여러 종류 메모리를 가짐:
//   - DEVICE_LOCAL: GPU 전용, 빠름, CPU 접근 불가
//   - HOST_VISIBLE: CPU에서 접근 가능, 느림
//   - HOST_COHERENT: CPU 쓰면 자동 동기화 (flush 불필요)
//
// properties 예시:
//   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  → GPU 전용 (shader용)
//   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  → CPU 쓰기 가능 (업로드용)
// ------------------------------------------------------------
inline uint32_t findMemoryType(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,              // 버퍼가 요구하는 메모리 타입 비트마스크
    VkMemoryPropertyFlags properties  // 우리가 원하는 속성
) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        bool typeMatches = (typeFilter & (1 << i)) != 0;
        bool propsMatch  = (memProps.memoryTypes[i].propertyFlags & properties) == properties;
        if (typeMatches && propsMatch) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

// ------------------------------------------------------------
// BufferBundle: VkBuffer + VkDeviceMemory 묶음
// ------------------------------------------------------------
// Vulkan에서 버퍼는 2단계:
//   1. VkBuffer 생성 (메타데이터만, 메모리 없음)
//   2. VkDeviceMemory 할당 후 바인딩
//
// 항상 짝으로 다니므로 묶어서 관리
// ------------------------------------------------------------
struct BufferBundle {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

// ------------------------------------------------------------
// createBuffer: 범용 버퍼 생성
// ------------------------------------------------------------
// usage 예시:
//   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  → SSBO (shader에서 읽기/쓰기)
//   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT  → UBO (상수 데이터)
//   VK_BUFFER_USAGE_TRANSFER_SRC_BIT    → staging (CPU→GPU 복사 출발지)
//   VK_BUFFER_USAGE_TRANSFER_DST_BIT    → GPU 버퍼 (복사 목적지)
// ------------------------------------------------------------
inline BufferBundle createBuffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memProps
) {
    BufferBundle bundle;
    bundle.size = size;

    // Step 1: Create buffer (metadata only)
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = size;
    bufferInfo.usage       = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;  // 한 큐에서만 사용

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &bundle.buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    // Step 2: Query memory requirements
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, bundle.buffer, &memReq);

    // Step 3: Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReq.memoryTypeBits, memProps);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bundle.memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }

    // Step 4: Bind memory to buffer
    vkBindBufferMemory(device, bundle.buffer, bundle.memory, 0);

    return bundle;
}

// ------------------------------------------------------------
// destroyBuffer: 버퍼 정리
// ------------------------------------------------------------
inline void destroyBuffer(VkDevice device, BufferBundle& bundle) {
    if (bundle.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, bundle.buffer, nullptr);
        bundle.buffer = VK_NULL_HANDLE;
    }
    if (bundle.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, bundle.memory, nullptr);
        bundle.memory = VK_NULL_HANDLE;
    }
}

// ------------------------------------------------------------
// uploadToBuffer: CPU 데이터 → GPU 버퍼 복사
// ------------------------------------------------------------
// 주의: HOST_VISIBLE 메모리에만 사용 가능
// DEVICE_LOCAL 버퍼는 staging buffer 경유해야 함
// ------------------------------------------------------------
inline void uploadToBuffer(
    VkDevice device,
    BufferBundle& bundle,
    const void* data,
    VkDeviceSize size
) {
    void* mapped;
    vkMapMemory(device, bundle.memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(device, bundle.memory);
}

// ------------------------------------------------------------
// downloadFromBuffer: GPU 버퍼 → CPU 복사
// ------------------------------------------------------------
inline void downloadFromBuffer(
    VkDevice device,
    BufferBundle& bundle,
    void* data,
    VkDeviceSize size
) {
    void* mapped;
    vkMapMemory(device, bundle.memory, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(device, bundle.memory);
}

} // namespace gs