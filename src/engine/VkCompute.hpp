// ============================================================
// File: src/engine/VkCompute.hpp (v2 - 다중 바인딩 + push constants)
// ============================================================
#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>

namespace gs {

// ------------------------------------------------------------
// SPIR-V 로드 (동일)
// ------------------------------------------------------------
inline std::vector<char> loadSPV(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// ------------------------------------------------------------
// ComputeContext (동일)
// ------------------------------------------------------------
struct ComputeContext {
    VkShaderModule        shaderModule        = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            pipeline            = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
};

// ------------------------------------------------------------
// createComputePipeline (확장: 바인딩 개수 + push constant 크기)
// ------------------------------------------------------------
// bindingCount: SSBO 개수 (binding 0, 1, 2, ...)
// pushConstantSize: push constant 구조체 크기 (0이면 안 씀)
//
// 예시:
//   simple.comp: bindingCount=1, pushConstantSize=0
//   gaussian.comp: bindingCount=2, pushConstantSize=12 (3 * uint)
// ------------------------------------------------------------
inline ComputeContext createComputePipeline(
    VkDevice device,
    const std::string& shaderPath,
    uint32_t bindingCount = 1,
    uint32_t pushConstantSize = 0
) {
    ComputeContext ctx;
    
    // ===== Shader Module =====
    auto code = loadSPV(shaderPath);
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &ctx.shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    printf("  [1/5] Shader module loaded\n");
    
    // ===== Descriptor Set Layout (다중 바인딩) =====
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; i++) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings    = bindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &ctx.descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    printf("  [2/5] Descriptor set layout (%u bindings)\n", bindingCount);
    
    // ===== Pipeline Layout (push constants 포함) =====
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts    = &ctx.descriptorSetLayout;
    
    VkPushConstantRange pushRange{};
    if (pushConstantSize > 0) {
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset     = 0;
        pushRange.size       = pushConstantSize;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges    = &pushRange;
    }
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &ctx.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    printf("  [3/5] Pipeline layout (push=%u bytes)\n", pushConstantSize);
    
    // ===== Compute Pipeline =====
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = ctx.pipelineLayout;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = ctx.shaderModule;
    pipelineInfo.stage.pName  = "main";
    
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ctx.pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    printf("  [4/5] Compute pipeline created\n");
    
    // ===== Descriptor Pool =====
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindingCount * 2;  // 여유분
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = 4;
    
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &ctx.descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
    printf("  [5/5] Descriptor pool created\n");
    
    // ===== Allocate Descriptor Set =====
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = ctx.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &ctx.descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &allocInfo, &ctx.descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    printf("  [+] Descriptor set allocated\n");
    
    return ctx;
}

// bindSSBO, destroyComputePipeline (동일)
inline void bindSSBO(VkDevice device, ComputeContext& ctx, VkBuffer buffer, VkDeviceSize size, uint32_t binding = 0) {
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range  = size;
    
    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = ctx.descriptorSet;
    write.dstBinding      = binding;
    write.dstArrayElement = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo     = &bufferInfo;
    
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

inline void destroyComputePipeline(VkDevice device, ComputeContext& ctx) {
    vkDestroyPipeline(device, ctx.pipeline, nullptr);
    vkDestroyPipelineLayout(device, ctx.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, ctx.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, ctx.descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, ctx.shaderModule, nullptr);
}

} // namespace gs