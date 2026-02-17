// ============================================================
// File: src/engine/VkEngine.hpp
// Role: Minimal Vulkan setup for compute shader execution
// Note: No swapchain, no graphics pipeline (reuse company code later)
// ============================================================
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <stdexcept>
#include <cstdio>

namespace gs {

class VkEngine {
public:
    // --------------------------------------------------------
    // Lifecycle
    // --------------------------------------------------------
    void init(GLFWwindow* window) {
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        allocateCommandBuffer();
        printf("[VkEngine] Initialized successfully\n");
    }

    void cleanup() {
        // Reverse order of creation
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        printf("[VkEngine] Cleaned up\n");
    }

    // --------------------------------------------------------
    // Getters (Phase 1+ will need these)
    // --------------------------------------------------------
    VkDevice       device()        const { return device_; }
    VkQueue        computeQueue()  const { return computeQueue_; }
    VkCommandPool  commandPool()   const { return commandPool_; }
    VkCommandBuffer commandBuffer() const { return commandBuffer_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }

private:
    // --------------------------------------------------------
    // Vulkan handles
    // --------------------------------------------------------
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkCommandBuffer  commandBuffer_  = VK_NULL_HANDLE;
    uint32_t         computeQueueFamily_ = 0;

    // --------------------------------------------------------
    // Step 1: Create Vulkan Instance
    // --------------------------------------------------------
    // Instance = Vulkan library connection
    // Think of it as: "Hey Vulkan, I exist!"
    // --------------------------------------------------------
    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "GaussianSplat";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName        = "NoEngine";
        appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion         = VK_API_VERSION_1_2;  // Vulkan 1.2 for compute

        VkInstanceCreateInfo createInfo{};
        createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // No extensions needed for pure compute (no surface/swapchain)
        createInfo.enabledExtensionCount   = 0;
        createInfo.enabledLayerCount       = 0;

        if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }
        printf("  [1/5] Instance created\n");
    }

    // --------------------------------------------------------
    // Step 2: Pick Physical Device (GPU)
    // --------------------------------------------------------
    // Physical Device = actual GPU hardware
    // We pick the first discrete GPU, or fallback to any GPU
    // --------------------------------------------------------
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("No Vulkan-capable GPU found");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        // Prefer discrete GPU (like RTX 1080)
        for (auto& dev : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice_ = dev;
                printf("  [2/5] GPU selected: %s (discrete)\n", props.deviceName);
                return;
            }
        }

        // Fallback: first available
        physicalDevice_ = devices[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        printf("  [2/5] GPU selected: %s (fallback)\n", props.deviceName);
    }

    // --------------------------------------------------------
    // Step 3: Create Logical Device + Compute Queue
    // --------------------------------------------------------
    // Logical Device = our interface to the GPU
    // Queue = where we submit commands
    //
    // Queue Family types:
    //   - Graphics: draw calls
    //   - Compute:  compute shaders  <-- we need this
    //   - Transfer: memory copy
    // --------------------------------------------------------
    void createLogicalDevice() {
        // Find compute queue family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, queueFamilies.data());

        bool found = false;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily_ = i;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("No compute queue family found");
        }

        // Queue creation info
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamily_;
        queueCreateInfo.queueCount       = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Device features (empty for now, add if needed)
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Create logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount    = 1;
        createInfo.pQueueCreateInfos       = &queueCreateInfo;
        createInfo.pEnabledFeatures        = &deviceFeatures;
        createInfo.enabledExtensionCount   = 0;  // No swapchain extension needed
        createInfo.enabledLayerCount       = 0;

        if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }

        // Get compute queue handle
        vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
        printf("  [3/5] Logical device + compute queue (family %u)\n", computeQueueFamily_);
    }

    // --------------------------------------------------------
    // Step 4: Create Command Pool
    // --------------------------------------------------------
    // Command Pool = memory pool for command buffers
    // Think of it as: notepad factory
    // --------------------------------------------------------
    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = computeQueueFamily_;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            // ^ Allows individual command buffer reset (useful for iterative compute)

        if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
        printf("  [4/5] Command pool created\n");
    }

    // --------------------------------------------------------
    // Step 5: Allocate Command Buffer
    // --------------------------------------------------------
    // Command Buffer = list of GPU commands
    // Think of it as: a to-do list we hand to the GPU
    //
    // Usage pattern:
    //   1. Begin recording
    //   2. Bind pipeline, dispatch compute
    //   3. End recording
    //   4. Submit to queue
    // --------------------------------------------------------
    void allocateCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = commandPool_;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }
        printf("  [5/5] Command buffer allocated\n");
    }
};

} // namespace gs