#include <compute/blur/scene.hpp>

#include <psemek/gfx/array.hpp>
#include <psemek/gfx/program.hpp>
#include <psemek/gfx/framebuffer.hpp>
#include <psemek/gfx/texture.hpp>
#include <psemek/gfx/renderbuffer.hpp>
#include <psemek/gfx/painter.hpp>
#include <psemek/gfx/error.hpp>
#include <psemek/gfx/query.hpp>
#include <psemek/geom/camera.hpp>
#include <psemek/util/clock.hpp>
#include <psemek/util/to_string.hpp>
#include <psemek/util/moving_average.hpp>

namespace compute
{

	namespace
	{

		char const compute_separable_lds_compact_horizontal_compute[] =
R"(#version 430

const int GROUP_SIZE = 64;

layout(local_size_x = 64, local_size_y = 1) in;
layout(r32ui, binding = 0) uniform restrict readonly uimage2D u_input_image;
layout(rgba8, binding = 1) uniform restrict writeonly image2D u_output_image;

const int M = 16;
const int N = 2 * M + 1;

// sigma = 10
const float coeffs[N] = float[N](
	0.012318109844189502,
	0.014381474814203989,
	0.016623532195728208,
	0.019024086115486723,
	0.02155484948872149,
	0.02417948052890078,
	0.02685404941667096,
	0.0295279624870386,
	0.03214534135442581,
	0.03464682117793548,
	0.0369716985390341,
	0.039060328279673276,
	0.040856643282313365,
	0.04231065439216247,
	0.043380781642569775,
	0.044035873841196206,
	0.04425662519949865,
	0.044035873841196206,
	0.043380781642569775,
	0.04231065439216247,
	0.040856643282313365,
	0.039060328279673276,
	0.0369716985390341,
	0.03464682117793548,
	0.03214534135442581,
	0.0295279624870386,
	0.02685404941667096,
	0.02417948052890078,
	0.02155484948872149,
	0.019024086115486723,
	0.016623532195728208,
	0.014381474814203989,
	0.012318109844189502
);

const int CACHE_SIZE = GROUP_SIZE + 2 * M;

const int LOAD = (CACHE_SIZE + (GROUP_SIZE - 1)) / GROUP_SIZE;

shared uint cache[CACHE_SIZE];

vec4 uint_to_vec4(uint x)
{
	return vec4(
		float((x & 0x000000ff) >>  0) / 255.0,
		float((x & 0x0000ff00) >>  8) / 255.0,
		float((x & 0x00ff0000) >> 16) / 255.0,
		float((x & 0xff000000) >> 24) / 255.0
	);
}

void main()
{
	ivec2 size = imageSize(u_input_image);
	ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);

	int origin = int(gl_WorkGroupID.x) * GROUP_SIZE - M;

	for (int i = 0; i < LOAD; ++i)
	{
		int local = int(gl_LocalInvocationID.x) * LOAD + i;
		if (local < CACHE_SIZE)
		{
			int pc = origin + local;

			if (pc >= 0 && pc < size.x)
				cache[local] = imageLoad(u_input_image, ivec2(pc, pixel_coord.y)).r;
		}
	}

	memoryBarrierShared();
	barrier();

	if (pixel_coord.x < size.x && pixel_coord.y < size.y)
	{
		vec4 sum = vec4(0.0);

		for (int i = 0; i < N; ++i)
		{
			ivec2 pc = pixel_coord + ivec2(i - M, 0);
			if (pc.x < 0) pc.x = 0;
			if (pc.x >= size.x) pc.x = size.x - 1;

			int local = pc.x - origin;

			sum += coeffs[i] * uint_to_vec4(cache[local]);
		}

		imageStore(u_output_image, pixel_coord, sum);
	}
}
)";

		char const compute_separable_lds_compact_vertical_compute[] =
R"(#version 430

const int GROUP_SIZE = 64;

layout(local_size_x = 1, local_size_y = 64) in;
layout(r32ui, binding = 0) uniform restrict readonly uimage2D u_input_image;
layout(rgba8, binding = 1) uniform restrict writeonly image2D u_output_image;

const int M = 16;
const int N = 2 * M + 1;

// sigma = 10
const float coeffs[N] = float[N](
	0.012318109844189502,
	0.014381474814203989,
	0.016623532195728208,
	0.019024086115486723,
	0.02155484948872149,
	0.02417948052890078,
	0.02685404941667096,
	0.0295279624870386,
	0.03214534135442581,
	0.03464682117793548,
	0.0369716985390341,
	0.039060328279673276,
	0.040856643282313365,
	0.04231065439216247,
	0.043380781642569775,
	0.044035873841196206,
	0.04425662519949865,
	0.044035873841196206,
	0.043380781642569775,
	0.04231065439216247,
	0.040856643282313365,
	0.039060328279673276,
	0.0369716985390341,
	0.03464682117793548,
	0.03214534135442581,
	0.0295279624870386,
	0.02685404941667096,
	0.02417948052890078,
	0.02155484948872149,
	0.019024086115486723,
	0.016623532195728208,
	0.014381474814203989,
	0.012318109844189502
);

const int CACHE_SIZE = GROUP_SIZE + 2 * M;

const int LOAD = (CACHE_SIZE + (GROUP_SIZE - 1)) / GROUP_SIZE;

shared uint cache[CACHE_SIZE];

vec4 uint_to_vec4(uint x)
{
	return vec4(
		float((x & 0x000000ff) >>  0) / 255.0,
		float((x & 0x0000ff00) >>  8) / 255.0,
		float((x & 0x00ff0000) >> 16) / 255.0,
		float((x & 0xff000000) >> 24) / 255.0
	);
}

void main()
{
	ivec2 size = imageSize(u_input_image);
	ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);

	int origin = int(gl_WorkGroupID.y) * GROUP_SIZE - M;

	for (int i = 0; i < LOAD; ++i)
	{
		int local = int(gl_LocalInvocationID.y) * LOAD + i;
		if (local < CACHE_SIZE)
		{
			int pc = origin + local;

			if (pc >= 0 && pc < size.y)
				cache[local] = imageLoad(u_input_image, ivec2(pixel_coord.x, pc)).r;
		}
	}

	memoryBarrierShared();
	barrier();

	if (pixel_coord.x < size.x && pixel_coord.y < size.y)
	{
		vec4 sum = vec4(0.0);

		for (int i = 0; i < N; ++i)
		{
			ivec2 pc = pixel_coord + ivec2(0, i - M);
			if (pc.y < 0) pc.y = 0;
			if (pc.y >= size.y) pc.y = size.y - 1;

			int local = pc.y - origin;

			sum += coeffs[i] * uint_to_vec4(cache[local]);
		}

		imageStore(u_output_image, pixel_coord, sum);
	}
}
)";

		struct compute_separable_lds_compact_impl
			: scene
		{
			compute_separable_lds_compact_impl();

			void on_resize(int width, int height) override;

			void present() override;

		private:
			util::clock<std::chrono::duration<float>, std::chrono::high_resolution_clock> clock_;

			gfx::framebuffer fbo_1_;
			gfx::texture_2d color_buffer_1_;
			gfx::renderbuffer depth_buffer_1_;

			gfx::framebuffer fbo_2_;
			gfx::texture_2d color_buffer_2_;

			gfx::framebuffer fbo_3_;
			gfx::texture_2d color_buffer_3_;

			gfx::program blur_horizontal_program_{compute_separable_lds_compact_horizontal_compute};
			gfx::program blur_vertical_program_{compute_separable_lds_compact_vertical_compute};

			gfx::painter painter_;

			gfx::query_pool queries_;

			util::moving_average<float> frame_time_{32};
			util::moving_average<float> blur_time_{32};
		};

		compute_separable_lds_compact_impl::compute_separable_lds_compact_impl()
		{
			color_buffer_1_.linear_filter();
			color_buffer_1_.clamp();

			color_buffer_2_.linear_filter();
			color_buffer_2_.clamp();

			color_buffer_3_.linear_filter();
			color_buffer_3_.clamp();
		}

		void compute_separable_lds_compact_impl::on_resize(int width, int height)
		{
			scene::on_resize(width, height);

			color_buffer_1_.load<gfx::color_rgba>({width, height});
			depth_buffer_1_.storage<gfx::depth24_pixel>({width, height});

			color_buffer_2_.load<gfx::color_rgba>({width, height});

			color_buffer_3_.load<gfx::color_rgba>({width, height});

			fbo_1_.color(color_buffer_1_);
			fbo_1_.depth(depth_buffer_1_);

			fbo_2_.color(color_buffer_2_);

			fbo_3_.color(color_buffer_3_);

			fbo_1_.assert_complete();
			fbo_2_.assert_complete();
			fbo_3_.assert_complete();
		}

		void compute_separable_lds_compact_impl::present()
		{
			float const dt = clock_.restart().count();
			frame_time_.push(dt);

			fbo_1_.bind();
			scene::draw();

			fbo_2_.bind();

			{
				auto scope = queries_.begin(gl::TIME_ELAPSED, [this](GLint value){ blur_time_.push(value / 1e6f); });

				gl::Clear(gl::COLOR_BUFFER_BIT);
				gl::Disable(gl::DEPTH_TEST);

				gl::MemoryBarrier(gl::SHADER_IMAGE_ACCESS_BARRIER_BIT);

				int const group_size = 64;

				blur_horizontal_program_.bind();

				gl::BindImageTexture(0, color_buffer_1_.id(), 0, gl::FALSE, 0, gl::READ_ONLY, gl::RGBA8);
				gl::BindImageTexture(1, color_buffer_2_.id(), 0, gl::FALSE, 0, gl::WRITE_ONLY, gl::RGBA8);
				gl::DispatchCompute((width() + group_size - 1) / group_size, height(), 1);

				gl::MemoryBarrier(gl::SHADER_IMAGE_ACCESS_BARRIER_BIT);

				blur_vertical_program_.bind();

				gl::BindImageTexture(0, color_buffer_2_.id(), 0, gl::FALSE, 0, gl::READ_ONLY, gl::RGBA8);
				gl::BindImageTexture(1, color_buffer_3_.id(), 0, gl::FALSE, 0, gl::WRITE_ONLY, gl::RGBA8);
				gl::DispatchCompute(width(), (height() + group_size - 1) / group_size, 1);

				gl::MemoryBarrier(gl::FRAMEBUFFER_BARRIER_BIT);
			}

			gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fbo_3_.id());
			gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
			gl::BlitFramebuffer(0, 0, width(), height(), 0, 0, width(), height(), gl::COLOR_BUFFER_BIT, gl::NEAREST);

			gfx::framebuffer::null().bind();

			{
				gfx::painter::text_options opts;
				opts.scale = 2.f;
				opts.c = gfx::black;
				opts.x = gfx::painter::x_align::left;
				opts.y = gfx::painter::y_align::top;

				painter_.text({20.f, 20.f}, "Compute separable LDS compact", opts);

				painter_.text({20.f, 40.f}, util::to_string("FPS: ", 1.f / frame_time_.average()), opts);

				if (blur_time_.count() > 0)
					painter_.text({20.f, 60.f}, util::to_string("Blur: ", blur_time_.average(), "ms"), opts);
			}

			painter_.render(geom::window_camera{width(), height()}.transform());

			queries_.poll();
		}

	}

	std::unique_ptr<scene> compute_separable_lds_compact()
	{
		if (!gl::sys::ext_ARB_compute_shader())
			throw std::runtime_error("OpenGL extension ARB_compute_shader not supported");

		if (!gl::sys::ext_ARB_shader_image_load_store())
			throw std::runtime_error("OpenGL extension ARB_shader_image_load_store not supported");

		return std::make_unique<compute_separable_lds_compact_impl>();
	}

}
