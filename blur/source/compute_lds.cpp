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

		char const compute_lds_compute[] =
R"(#version 430

const int GROUP_SIZE = 16;

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform restrict readonly image2D u_input_image;
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

shared vec4 cache[GROUP_SIZE + 2 * M][GROUP_SIZE + 2 * M];

const int LOAD = (GROUP_SIZE + 2 * M) / GROUP_SIZE;

void main()
{
//	ivec2 u_direction = ivec2(1, 0);

	ivec2 size = imageSize(u_input_image);
	ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);

	ivec2 workgroup_origin = ivec2(gl_WorkGroupID.xy) * GROUP_SIZE - ivec2(M, M);

	// Populate shared group cache
	for (int i = 0; i < LOAD; ++i)
	{
		for (int j = 0; j < LOAD; ++j)
		{
			ivec2 local = ivec2(gl_LocalInvocationID.xy) * LOAD + ivec2(i, j);
			ivec2 pc = workgroup_origin + local;

			if (pc.x >= 0 && pc.y >= 0 && pc.x < size.x && pc.y < size.y)
			{
				cache[local.x][local.y] = imageLoad(u_input_image, pc);
			}
		}
	}

	memoryBarrierShared();
	barrier();

	if (pixel_coord.x < size.x && pixel_coord.y < size.y)
	{
		vec4 sum = vec4(0.0);

		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				ivec2 pc = pixel_coord + ivec2(i - M, j - M);
				if (pc.x < 0) pc.x = 0;
				if (pc.y < 0) pc.y = 0;
				if (pc.x >= size.x) pc.x = size.x - 1;
				if (pc.y >= size.y) pc.y = size.y - 1;

				ivec2 local = pc - workgroup_origin;

				sum += coeffs[i] * coeffs[j] * cache[local.x][local.y];
			}
		}

		imageStore(u_output_image, pixel_coord, sum);
	}
}
)";

		struct compute_lds_impl
			: scene
		{
			compute_lds_impl();

			void on_resize(int width, int height) override;

			void present() override;

		private:
			util::clock<std::chrono::duration<float>, std::chrono::high_resolution_clock> clock_;

			gfx::framebuffer fbo_1_;
			gfx::texture_2d color_buffer_1_;
			gfx::renderbuffer depth_buffer_1_;

			gfx::framebuffer fbo_2_;
			gfx::texture_2d color_buffer_2_;

			gfx::program blur_program_{compute_lds_compute};

			gfx::painter painter_;

			gfx::query_pool queries_;

			util::moving_average<float> frame_time_{32};
			util::moving_average<float> blur_time_{32};
		};

		compute_lds_impl::compute_lds_impl()
		{
			color_buffer_1_.linear_filter();
			color_buffer_1_.clamp();

			color_buffer_2_.linear_filter();
			color_buffer_2_.clamp();
		}

		void compute_lds_impl::on_resize(int width, int height)
		{
			scene::on_resize(width, height);

			color_buffer_1_.load<gfx::color_rgba>({width, height});
			depth_buffer_1_.storage<gfx::depth24_pixel>({width, height});

			color_buffer_2_.load<gfx::color_rgba>({width, height});

			fbo_1_.color(color_buffer_1_);
			fbo_1_.depth(depth_buffer_1_);

			fbo_2_.color(color_buffer_2_);

			fbo_1_.assert_complete();
			fbo_2_.assert_complete();
		}

		void compute_lds_impl::present()
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

				int const group_size = 16;

				blur_program_.bind();
				gl::BindImageTexture(0, color_buffer_1_.id(), 0, gl::FALSE, 0, gl::READ_ONLY, gl::RGBA8);
				gl::BindImageTexture(1, color_buffer_2_.id(), 0, gl::FALSE, 0, gl::WRITE_ONLY, gl::RGBA8);
				gl::DispatchCompute((width() + group_size - 1) / group_size, (height() + group_size - 1) / group_size, 1);

				gl::MemoryBarrier(gl::FRAMEBUFFER_BARRIER_BIT);
			}

			gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fbo_2_.id());
			gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
			gl::BlitFramebuffer(0, 0, width(), height(), 0, 0, width(), height(), gl::COLOR_BUFFER_BIT, gl::NEAREST);

			gfx::framebuffer::null().bind();

			{
				gfx::painter::text_options opts;
				opts.scale = 2.f;
				opts.c = gfx::black;
				opts.x = gfx::painter::x_align::left;
				opts.y = gfx::painter::y_align::top;

				painter_.text({20.f, 20.f}, "Compute LDS", opts);

				painter_.text({20.f, 40.f}, util::to_string("FPS: ", 1.f / frame_time_.average()), opts);

				if (blur_time_.count() > 0)
					painter_.text({20.f, 60.f}, util::to_string("Blur: ", blur_time_.average(), "ms"), opts);
			}

			painter_.render(geom::window_camera{width(), height()}.transform());

			queries_.poll();
		}

	}

	std::unique_ptr<scene> compute_lds()
	{
		if (!gl::sys::ext_ARB_compute_shader())
			throw std::runtime_error("OpenGL extension ARB_compute_shader not supported");

		if (!gl::sys::ext_ARB_shader_image_load_store())
			throw std::runtime_error("OpenGL extension ARB_shader_image_load_store not supported");

		return std::make_unique<compute_lds_impl>();
	}

}
