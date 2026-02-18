
gaussian-splat
- src
    - engine
        - VkBuffer.hpp
            - inline uint32_t findMemoryType(
                            VkPhysicalDevice physicalDevice,
                            uint32_t typeFilter,              // 버퍼가 요구하는 메모리 타입 비트마스크
                            VkMemoryPropertyFlags properties  // 우리가 원하는 속성
                        )
            - struct BufferBundle {
                            VkBuffer       buffer = VK_NULL_HANDLE;
                            VkDeviceMemory memory = VK_NULL_HANDLE;
                            VkDeviceSize   size   = 0;
                        };
            - inline BufferBundle createBuffer(
                            VkDevice device,
                            VkPhysicalDevice physicalDevice,
                            VkDeviceSize size,
                            VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags memProps
                        )
            - inline void destroyBuffer(VkDevice device, BufferBundle& bundle)
            - inline void uploadToBuffer(
                            VkDevice device,
                            BufferBundle& bundle,
                            const void* data,
                            VkDeviceSize size
                        )
            - inline void downloadFromBuffer(
                            VkDevice device,
                            BufferBundle& bundle,
                            void* data,
                            VkDeviceSize size
                        )
        - VkCompute.hpp
            - inline std::vector<char> loadSPV(const std::string& filename)
            - struct ComputeContext {
                    VkShaderModule        shaderModule        = VK_NULL_HANDLE;
                    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
                    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
                    VkPipeline            pipeline            = VK_NULL_HANDLE;
                    VkDescriptorPool      descriptorPool      = VK_NULL_HANDLE;
                    VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
                };
            - inline ComputeContext createComputePipeline(
                    VkDevice device,
                    const std::string& shaderPath,
                    uint32_t bindingCount = 1,
                    uint32_t pushConstantSize = 0
                )
            - inline void bindSSBO(VkDevice device, ComputeContext& ctx, VkBuffer buffer, 
                    VkDeviceSize size, uint32_t binding = 0) 
            - inline void destroyComputePipeline(VkDevice device, ComputeContext& ctx)
        - VkEngine.hpp
            -     void init(GLFWwindow* window) {
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
    - shaders
        - backward.comp
        - gaussian.comp
        - loss.comp
        - simple.comp
    - utils
        - ImageIO.hpp
            - inline bool savePPM(
                const std::string& filename,
                const std::vector<glm::vec4>& pixels,
                uint32_t width,
                uint32_t height
            )
    - main.cpp
        - struct RenderPC {
            uint32_t width;
            uint32_t height;
            uint32_t gaussCount;
        };

        struct LossPC {
            uint32_t width;
            uint32_t height;
        };

        // ============================================================
        // GaussianGrad
        // ============================================================
        struct GaussianGrad {
            glm::vec3 dPosition;
            float     dOpacity;
            glm::vec3 dScale;
            float     _pad0;
            glm::vec4 dRotation;
            glm::vec3 dColor;
            float     _pad1;
        };
        static_assert(sizeof(GaussianGrad) == 64, "GaussianGrad must be 64 bytes");

        // ============================================================
        // CPU에서 가우시안 렌더링 (target 생성용)
        // ============================================================
        void renderGaussiansCPU(
            std::vector<glm::vec4>& pixels,
            const std::vector<gs::GaussianParam>& gaussians,
            uint32_t width, uint32_t height
        ) 
        main function
        - GLFW + Vulkan
        - Pipelines 
            - createComputePipeline (gaussian.spv)
            - createComputePipeline (loss.spv)
        - target gaussians
        - learnable gaussians
        - create buffers
        - descriptor binding (bindSSBO)
        - train loop (for loop until MAX_ITER)
            - parameter upload
            - command buffer
            - forward
            - barrier
            - loss
            - submit
            - accumulate loss
            - calculate gradient on cpu
        - log
        