use core::{ffi, time};
use std::{
    ffi::{CStr, CString},
    fs,
    mem::offset_of,
    ops::Deref,
    thread,
};

use tracing::{debug, info, Instrument};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

use ash::{
    Device, Entry, Instance,
    khr::{surface, swapchain},
    prelude::VkResult,
    util::read_spv,
    vk::{
        self, Buffer, BufferUsageFlags, CommandBuffer, CommandPool, ComponentMapping,
        DescriptorPool, DescriptorSet, DescriptorSetLayout, DeviceMemory, DeviceSize, DynamicState,
        Extent2D, Fence, Format, Framebuffer, Image, ImageSubresourceRange, ImageView, Offset2D,
        PhysicalDevice, Pipeline, PipelineLayout, Queue, Rect2D, RenderPass, Semaphore,
        ShaderModule, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
    },
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

struct Renderer {
    instance: Instance, // the actual vulkan instance
    entry: Entry, // object used for looking up and executing the vulkan library functions, this thing is parent to the instance
    physical_device: PhysicalDevice,
    device: Device,
    queue: Queue,
    queue_family_index: u32,
    surface: SurfaceKHR,
    swapchain: SwapchainKHR,
    swapchain_loader: swapchain::Device, // used to acquire images from the swapchain
    swapchain_images: Vec<Image>,
    swapchain_extent: Extent2D,
    swapchain_format: Format,
    swapchain_image_views: Vec<ImageView>,
    render_pass: RenderPass,
    graphics_pipeline: Pipeline,
    graphics_pipeline_layout: PipelineLayout,
    framebuffers: Vec<Framebuffer>,
    command_pool: CommandPool,
    command_buffers: Vec<CommandBuffer>,
    vertex_buffer: Buffer,
    uniform_buffers: Vec<Buffer>,
    uniform_memories: Vec<DeviceMemory>,
    uniform_handles: Vec<*mut ffi::c_void>,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
    image_available_semaphores: Vec<Semaphore>,
    render_finished_semaphores: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,
    current_frame: u32,
    framebuffer_resized: bool,
}

#[derive(Debug)]
#[repr(C)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

#[derive(Debug)]
#[repr(C)]
struct Uniform {
    looped: f32, // Value between 0 and 1 which wraps every second
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);

        binding_description
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        let position_description = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32);

        let color_description = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32);

        let descriptions = vec![position_description, color_description];

        descriptions
    }
}

impl Renderer {
    fn run(&mut self) {}

    fn mainLoop(&mut self) {
        todo!();
    }

    fn cleanup(&mut self) {}

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = c"Ashes to ashes";

        let appInfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap();
        let mut extension_names = extension_names.to_vec();

        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());

        let props = unsafe { entry.enumerate_instance_layer_properties() }.unwrap();

        let layer_names = props
            .iter()
            .map(|p| p.layer_name_as_c_str().unwrap().as_ptr())
            .collect::<Vec<*const i8>>();

        let createInfo = vk::InstanceCreateInfo::default()
            .application_info(&appInfo)
            //.enabled_layer_names(layer_names.as_slice())
            .enabled_extension_names(&extension_names)
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);

        unsafe { entry.create_instance(&createInfo, None) }.unwrap()
    }

    fn pick_physical_device(instance: &Instance) -> PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let device: PhysicalDevice = devices
            .iter()
            .filter(|d| {
                let properties = unsafe { instance.get_physical_device_properties(**d) };
                let features = unsafe { instance.get_physical_device_features(**d) };

                properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                    && features.robust_buffer_access == 1 // Just arbitrary things to play around with device queries
            })
            .next()
            .expect("No suitable device found!!")
            .to_owned();

        device
    }

    fn find_graphical_queue_family_indices(
        instance: &Instance,
        device: &PhysicalDevice,
    ) -> Vec<u32> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };
        let queue_family_indices = queue_family_properties
            .iter()
            .enumerate()
            .filter(|(_, f)| f.queue_flags.intersects(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32)
            .collect::<Vec<u32>>();

        queue_family_indices
    }

    fn get_queue_family_index(instance: &Instance, device: &PhysicalDevice) -> u32 {
        let indices = Self::find_graphical_queue_family_indices(instance, device);
        let family_index = *indices.first().expect("no queue families were found");

        return family_index;
    }

    fn create_logical_device(instance: &Instance, device: &PhysicalDevice) -> (Device, Queue, u32) {
        let family_index = Self::get_queue_family_index(instance, device);

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_index)
            .queue_priorities(&[1.0]);

        let device_features = vk::PhysicalDeviceFeatures::default();

        let extension_names = vec![
            // This must be enabled because of moltenvk
            vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr(),
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
        ];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&extension_names)
            .enabled_features(&device_features);

        let logical_device =
            unsafe { instance.create_device(*device, &device_create_info, None) }.unwrap();

        // We just default to using the first (0th) queue in the queue fam
        let queue = unsafe { logical_device.get_device_queue(family_index, 0) };

        (logical_device, queue, family_index)
    }

    fn create_surface(window: &Window, entry: &Entry, instance: &Instance) -> SurfaceKHR {
        let display_handle = window.display_handle().unwrap().as_raw();
        let window_handle = window.window_handle().unwrap().as_raw();

        let surface = unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)
        }
        .unwrap();

        surface
    }

    fn create_swapchain(
        entry: &Entry,
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        surface: &SurfaceKHR,
    ) -> (
        SwapchainKHR,
        swapchain::Device,
        Vec<Image>,
        Extent2D,
        Format,
    ) {
        let surface_loader = surface::Instance::new(&entry, &instance);

        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let surface_format = unsafe {
            surface_loader.get_physical_device_surface_formats(*physical_device, *surface)
        }
        .unwrap()[0];

        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(*physical_device, *surface)
        }
        .unwrap();

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(*physical_device, *surface)
        }
        .unwrap();

        if !present_modes.contains(&vk::PresentModeKHR::FIFO) {
            panic!("FIFO not supported as present mode on surface!");
        }

        let present_mode = vk::PresentModeKHR::FIFO;

        // We request at least one more image, so we're not waiting around for
        // the driver to give us another image
        let image_count = surface_capabilities.min_image_count + 1;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1) // always 1 unless developing with stereoscopic 3D
            // color attachment usage means that we're rendering directly to the images
            // in the swapchain
            // it's possible to render to different buffers first, and then copy those
            // images into the swapchain. in which case the swapchain images will be
            // "transfer destinations"
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            // We might want to adjust this if we're using multiple queues for rendering
            // and presenting to the surface
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            // We can apply different transforms, like rotating by 90 degrees
            // for now, we'll just use whatever the surface expects
            .pre_transform(surface_capabilities.current_transform)
            // We don't do any alpha blending with other windows
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            // Clipping means that we don't care about what our pixels look like
            // when they're obscured by another window
            .clipped(true);

        let swapchain =
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }.unwrap();

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        (
            swapchain,
            swapchain_loader,
            images,
            surface_capabilities.current_extent,
            surface_format.format,
        )
    }

    fn get_image_views(format: &Format, images: &Vec<Image>, device: &Device) -> Vec<ImageView> {
        images
            .iter()
            .map(|i| {
                let img_create_info = vk::ImageViewCreateInfo::default()
                    .format(*format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(*i)
                    .components(ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1, // useful for XR applications, you could render to a layer per eye
                    });

                unsafe { device.create_image_view(&img_create_info, None) }.unwrap()
            })
            .collect::<Vec<ImageView>>()
            .to_owned()
    }

    fn create_shader_module(shader_words: &Vec<u32>, device: &Device) -> ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(shader_words);

        let shader_module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

        shader_module
    }

    fn create_render_pass(format: &Format, device: &Device) -> RenderPass {
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let dependencies = vec![dependency];

        let color_attachment = vk::AttachmentDescription::default()
            .format(*format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            // we don't care about what the frame looks like before writing
            // we clear the image anyway
            .initial_layout(vk::ImageLayout::UNDEFINED)
            // we do care about the format that the image is in after
            // rendering. PRESENT_SRC_KHR is a layout optimized for
            // displaying to the swapchain
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            // The index corresponds to the layout(location = 0) in our
            // fragment shader. The fragment shader references buffers
            // through the attachment mechanism
            .attachment(0)
            // We tell vulkan we want this attachment to be optimized
            // as a color attachment when we use it in our subpass
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let attachment_refs = vec![color_attachment_ref];
        let attachments = vec![color_attachment];

        let subpass = vk::SubpassDescription::default()
            // vulkan might support compute subpasses in the future
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            // This subpass uses one attachment, available as output
            // to location = 0 in the fragment shader
            .color_attachments(attachment_refs.as_slice());

        let subpasses = vec![subpass];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(attachments.as_slice())
            .subpasses(subpasses.as_slice())
            .dependencies(dependencies.as_slice());

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap();

        render_pass
    }

    fn create_descriptor_layout(device: &Device) -> DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = [ubo_layout_binding];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();

        descriptor_set_layout
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: &RenderPass,
        descriptor_set_layout: &DescriptorSetLayout,
    ) -> (Pipeline, PipelineLayout) {
        // Shaders

        let mut vertShaderFile = fs::File::open("vert.spv").unwrap();
        let vertShaderWords = read_spv(&mut vertShaderFile).unwrap();
        let mut fragShaderFile = fs::File::open("frag.spv").unwrap();
        let fragShaderWords = read_spv(&mut fragShaderFile).unwrap();

        let vertShaderModule = Self::create_shader_module(&vertShaderWords, device);
        let fragShaderModule = Self::create_shader_module(&fragShaderWords, device);

        let vertShaderStageInfo = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertShaderModule)
            .name(c"main");

        let fragShaderInfo = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragShaderModule)
            .name(c"main");

        let shader_stages = vec![vertShaderStageInfo, fragShaderInfo];

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions.as_slice());

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Viewport & Scissor

        let dynamic_states = vec![DynamicState::VIEWPORT, DynamicState::SCISSOR];

        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states.as_slice());

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterizer

        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.)
            .depth_bias_clamp(0.)
            .depth_bias_slope_factor(0.);

        // Multisampling

        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        // Color blending

        // This configures color blending per framebuffer attached to the pipeline
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        // This is global color blending settings for all framebuffers, here you could
        // set global constants that can be used by all framebuffers for their blending
        // calculations
        let color_blend_attachments = vec![color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(color_blend_attachments.as_slice());

        // Pipeline layout

        let layouts = [*descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

        // Creating the pipeline

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(shader_stages.as_slice())
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let pipeline_infos = vec![pipeline_info];

        let pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                pipeline_infos.as_slice(),
                None,
            )
        }
        .unwrap()
        .first()
        .unwrap()
        .to_owned();

        (pipeline, pipeline_layout)
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &Vec<ImageView>,
        render_pass: &RenderPass,
        extent: &Extent2D,
    ) -> Vec<Framebuffer> {
        image_views
            .iter()
            .map(|v| {
                let attachments = vec![*v];

                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(*render_pass)
                    .attachments(attachments.as_slice())
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None) }.unwrap()
            })
            .collect::<Vec<Framebuffer>>()
    }

    fn create_command_pool(device: &Device, family_queue_index: u32) -> CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(family_queue_index);

        unsafe { device.create_command_pool(&pool_info, None) }.unwrap()
    }

    fn create_command_buffers(device: &Device, command_pool: &CommandPool) -> Vec<CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            // you can call secondary command buffers from primary ones
            // almost like an abstraction over procedures
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap();

        command_buffers
    }

    fn create_sync_objects(device: &Device) -> (Vec<Semaphore>, Vec<Semaphore>, Vec<Fence>) {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // Create a CPU fence, and make sure to mark it as signaled
        // That way in our draw function, we can avoid having to
        // wait on the fence, since it's already signaled on the
        // first call
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores: Vec<Semaphore> = Vec::new();
        let mut render_finished_semaphores: Vec<Semaphore> = Vec::new();
        let mut in_flight_fences: Vec<Fence> = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap());
            render_finished_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap());
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None) }.unwrap());
        }

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    fn find_memory_type(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };

        let (idx, _property) = mem_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, m)| {
                let right_filter = (type_filter & (1 << i)) >= 1;
                let right_property = (m.property_flags & properties) == properties;

                right_filter && right_property
            })
            .expect("failed to find a suitable memory type!");

        idx as u32
    }

    fn create_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
        size: DeviceSize,
        usage_flags: BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (Buffer, DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.unwrap();

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type_idx = Self::find_memory_type(
            instance,
            physical_device,
            memory_requirements.memory_type_bits,
            properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_idx);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }.unwrap();

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.unwrap();

        (buffer, memory)
    }

    fn copy_buffer(
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        command_pool: &CommandPool,
        queue: &Queue,
        device: &Device,
        size: DeviceSize,
    ) {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(*command_pool)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap();

        let command_buffer = *command_buffers.first().unwrap();

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();

        let copy_region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size);

        let regions = [copy_region];

        unsafe { device.cmd_copy_buffer(command_buffer, *src_buffer, *dst_buffer, &regions) };

        let command_buffers = [command_buffer];

        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        let submits = [submit_info];

        unsafe { device.end_command_buffer(command_buffer) }.unwrap();

        unsafe { device.queue_submit(*queue, &submits, vk::Fence::null()) }.unwrap();

        unsafe { device.device_wait_idle() }.unwrap();

        unsafe { device.free_command_buffers(*command_pool, &command_buffers) };
    }

    fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
        command_pool: &CommandPool,
        queue: &Queue,
        vertices: &Vec<Vertex>,
    ) -> Buffer {
        let buffer_size = (size_of::<Vertex>() * vertices.len()) as DeviceSize;

        let (staging_buffer, staging_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let data = unsafe {
            device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
        }
        .unwrap();

        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                data as *mut Vertex,
                size_of::<Vertex>() as usize,
            )
        };

        unsafe {
            device.unmap_memory(staging_memory);
        };

        let (vertex_buffer, vertex_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            &staging_buffer,
            &vertex_buffer,
            command_pool,
            queue,
            device,
            buffer_size,
        );

        unsafe { device.destroy_buffer(staging_buffer, None) };

        unsafe { device.free_memory(staging_memory, None) };

        vertex_buffer
    }

    fn create_uniform_buffers(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
    ) -> (Vec<Buffer>, Vec<DeviceMemory>, Vec<*mut ffi::c_void>) {
        let buffer_size = size_of::<Uniform>();

        let buffer_info: (Vec<Buffer>, Vec<(DeviceMemory, *mut ffi::c_void)>) = (0
            ..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                let (buffer, memory) = Self::create_buffer(
                    instance,
                    device,
                    physical_device,
                    buffer_size as u64,
                    BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );

                let handle_into_buffer_memory = unsafe {
                    device.map_memory(memory, 0, buffer_size as u64, vk::MemoryMapFlags::empty())
                }
                .unwrap();

                (buffer, (memory, handle_into_buffer_memory))
            })
            .unzip();

        let buffer = buffer_info.0;
        let (memories, handles): (Vec<DeviceMemory>, Vec<*mut ffi::c_void>) =
            buffer_info.1.into_iter().unzip();

        (buffer, memories, handles)
    }

    fn create_descriptor_pool(device: &Device) -> DescriptorPool {
        let pool_size = vk::DescriptorPoolSize::default()
            .descriptor_count(MAX_FRAMES_IN_FLIGHT)
            .ty(vk::DescriptorType::UNIFORM_BUFFER);

        let pool_sizes = [pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.unwrap();

        pool
    }

    fn create_descriptor_sets(
        device: &Device,
        descriptor_pool: &DescriptorPool,
        descriptor_set_layout: &DescriptorSetLayout,
        uniform_buffers: &Vec<Buffer>,
    ) -> Vec<DescriptorSet> {
        let layouts: Vec<DescriptorSetLayout> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| descriptor_set_layout.clone())
            .collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*descriptor_pool)
            .set_layouts(layouts.as_slice());

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap();

        for i in (0..MAX_FRAMES_IN_FLIGHT) {
            let uniform_size = size_of::<Uniform>();

            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i as usize])
                .offset(0)
                .range(uniform_size as DeviceSize);

            let buffer_infos = [buffer_info];

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i as usize])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&buffer_infos);

            let descriptor_writes = [descriptor_write];

            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &[]);
            };
        }

        descriptor_sets
    }

    fn new(window: &Window) -> Self {
        let vertices = vec![
            Vertex {
                pos: [-0.5, -0.5],
                color: [1., 0., 0.],
            },
            Vertex {
                pos: [-0.25, 0.],
                color: [0., 1., 0.],
            },
            Vertex {
                pos: [-0.75, 0.],
                color: [0., 0., 1.],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [1., 0., 0.],
            },
            Vertex {
                pos: [0.75, 0.],
                color: [0., 0., 1.],
            },
            Vertex {
                pos: [0.25, 0.],
                color: [0., 1., 0.],
            },
        ];

        let entry = unsafe { Entry::load() }.expect("failed to create instance!");
        let instance = Self::create_instance(&entry, window);
        let physical_device = Self::pick_physical_device(&instance);
        let (device, queue, queue_family_index) =
            Self::create_logical_device(&instance, &physical_device);

        let surface = Self::create_surface(window, &entry, &instance);
        let (swapchain, swapchain_loader, swapchain_images, swapchain_extent, swapchain_format) =
            Self::create_swapchain(&entry, &instance, &physical_device, &device, &surface);
        let swapchain_image_views =
            Self::get_image_views(&swapchain_format, &swapchain_images, &device);

        let render_pass = Self::create_render_pass(&swapchain_format, &device);
        let descriptor_layout = Self::create_descriptor_layout(&device);
        let (graphics_pipeline, graphics_pipeline_layout) =
            Self::create_graphics_pipeline(&device, &render_pass, &descriptor_layout);
        let framebuffers = Self::create_framebuffers(
            &device,
            &swapchain_image_views,
            &render_pass,
            &swapchain_extent,
        );

        let command_pool = Self::create_command_pool(&device, queue_family_index);
        let vertex_buffer = Self::create_vertex_buffer(
            &instance,
            &device,
            &physical_device,
            &command_pool,
            &queue,
            &vertices,
        );

        let (uniform_buffers, uniform_memories, uniform_handles) =
            Self::create_uniform_buffers(&instance, &device, &physical_device);

        let descriptor_pool = Self::create_descriptor_pool(&device);
        let descriptor_sets = Self::create_descriptor_sets(
            &device,
            &descriptor_pool,
            &descriptor_layout,
            &uniform_buffers,
        );

        let command_buffers = Self::create_command_buffers(&device, &command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device);

        Self {
            instance,
            entry,
            physical_device,
            device,
            queue,
            queue_family_index,
            surface,
            swapchain,
            swapchain_loader,
            swapchain_images,
            swapchain_extent,
            swapchain_format,
            swapchain_image_views,
            render_pass,
            graphics_pipeline,
            graphics_pipeline_layout,
            framebuffers,
            command_pool,
            command_buffers,
            vertex_buffer,
            uniform_buffers,
            uniform_memories,
            uniform_handles,
            descriptor_pool,
            descriptor_sets,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            framebuffer_resized: false,
        }
    }

    fn record_command_buffer(&self, command_buffer: CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        // Start recording commands

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .unwrap();

        // Begin our render pass

        let mut clear_value = vk::ClearValue::default();
        clear_value.color = vk::ClearColorValue {
            float32: [0., 0., 0., 1.],
        };

        let clear_values = vec![clear_value];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(
                Rect2D::default()
                    .offset(Offset2D::default().x(0).y(0))
                    .extent(self.swapchain_extent),
            )
            .clear_values(clear_values.as_slice());

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE, // no secondary command buffers will be used
            )
        };

        // Bind our pipeline to our render pass

        // Note, we set up our scissor state and viewport to be dynamic, so we have to
        // set them up here

        let viewport = vk::Viewport::default()
            .x(0.)
            .y(0.)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.)
            .max_depth(1.);

        let viewports = vec![viewport];

        unsafe {
            self.device
                .cmd_set_viewport(command_buffer, 0, viewports.as_slice())
        };

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default().x(0).y(0))
            .extent(self.swapchain_extent);

        let scissors = vec![scissor];

        unsafe {
            self.device
                .cmd_set_scissor(command_buffer, 0, scissors.as_slice())
        };

        // Need to bind the pipeline

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        // Bind our descriptor sets

        let descriptor_sets = [self.descriptor_sets[self.current_frame as usize]];

        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline_layout,
                0,
                &descriptor_sets,
                &[]
            )
        };

        // Bind our vertex buffer

        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0 as u64];

        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets)
        };

        // Now we can issue the draw command for our triangles

        unsafe { self.device.cmd_draw(command_buffer, 6, 1, 0, 0) };

        // Now we can end the render pass

        unsafe { self.device.cmd_end_render_pass(command_buffer) };

        unsafe { self.device.end_command_buffer(command_buffer) };
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| self.device.destroy_framebuffer(*f, None));

            self.swapchain_image_views
                .iter()
                .for_each(|v| self.device.destroy_image_view(*v, None));

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn recreate_swapchain(&mut self) {
        // wait until the device finishes up any current work
        unsafe {
            self.device.device_wait_idle();
        };

        self.cleanup_swapchain();

        let (swapchain, swapchain_loader, images, extent, format) = Self::create_swapchain(
            &self.entry,
            &self.instance,
            &self.physical_device,
            &self.device,
            &self.surface,
        );

        let image_views = Self::get_image_views(&format, &images, &self.device);
        let framebuffers =
            Self::create_framebuffers(&self.device, &image_views, &self.render_pass, &extent);

        self.swapchain = swapchain;
        self.swapchain_loader = swapchain_loader;
        self.swapchain_images = images;
        self.swapchain_extent = extent;
        self.swapchain_format = format;
        self.swapchain_image_views = image_views;
        self.framebuffers = framebuffers;
    }

    fn update_uniforms(&mut self, start_time: std::time::Instant) {
        let now = std::time::Instant::now();
        let since_start = now - start_time;
        let looped = ((since_start.subsec_millis() as f32) / 1000.) as f32;

        let new_uniform = Uniform { looped };
        let uniform_size = size_of::<Uniform>();

        unsafe {
            std::ptr::copy_nonoverlapping(
                &new_uniform as *const Uniform,
                self.uniform_handles[self.current_frame as usize] as *mut Uniform,
                uniform_size,
            )
        };
    }

    // Returns whether the swapchain needs to be recreated
    fn draw_frame(&mut self, start_of_program: std::time::Instant) -> bool {
        debug!("Starting to wait on fence");
        // First thing we do is wait for our GPU to finish recieving our command buffer
        let fences = vec![self.in_flight_fences[self.current_frame as usize]];
        unsafe {
            self.device
                .wait_for_fences(fences.as_slice(), true, u64::MAX)
        }
        .unwrap();

        debug!("Acquiring image from swapchain");
        // Now we can acquire an image from the swapchain
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame as usize],
                Fence::null(),
            )
        };

        let image_idx = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return true;
            }
            Err(error) => panic!(
                "Error occurred while acquiring next image. Cause: {}",
                error
            ),
        };

        unsafe { self.device.reset_fences(fences.as_slice()) }.unwrap();

        // Make sure our command buffer is reset

        unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame as usize],
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .unwrap();

        debug!("Streaming commands into buffer");
        // Stream commands into our buffer

        self.record_command_buffer(
            self.command_buffers[self.current_frame as usize],
            image_idx as usize,
        );

        // Now we submit the command buffer

        let wait_semaphores = vec![self.image_available_semaphores[self.current_frame as usize]];
        let wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = vec![self.command_buffers[self.current_frame as usize]];
        let signal_semaphores = vec![self.render_finished_semaphores[self.current_frame as usize]];

        self.update_uniforms(start_of_program);

        let submit_debug = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores.as_slice())
            .wait_dst_stage_mask(wait_stages.as_slice())
            .command_buffers(command_buffers.as_slice())
            .signal_semaphores(signal_semaphores.as_slice());

        let submits = vec![submit_debug];

        debug!("Submitting commands to buffer");
        unsafe {
            self.device.queue_submit(
                self.queue,
                submits.as_slice(),
                self.in_flight_fences[self.current_frame as usize],
            );
        }

        // Now we present our image to the screen

        let swapchains = vec![self.swapchain];
        let image_indices = vec![image_idx];

        let present_debug = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores.as_slice())
            .swapchains(swapchains.as_slice())
            .image_indices(image_indices.as_slice());

        debug!("Queuing frame for presentation");

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_debug)
        }
        .unwrap();

        debug!("Finishing draw_frame");
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        false
    }
}

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    start_time: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vulkan tutorial with Ash")
                        .with_decorations(true)
                        .with_min_inner_size(PhysicalSize::new(1920, 1080)),
                )
                .unwrap(),
        );
        self.renderer = Some(Renderer::new(self.window.as_ref().unwrap()));
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        let mut renderer = self.renderer.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                renderer.framebuffer_resized = true;
            }
            WindowEvent::RedrawRequested => {
                if (renderer.draw_frame(self.start_time) || renderer.framebuffer_resized) {
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        debug!("Recreating swapchain");
                        renderer.framebuffer_resized = false;
                        renderer.recreate_swapchain();
                    } else {
                        return;
                    }
                }

                window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App {
        window: None,
        renderer: None,
        start_time: std::time::Instant::now(),
    };

    event_loop.run_app(&mut app).unwrap();
}
