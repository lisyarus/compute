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

		char const separable_vertex[] =
R"(#version 330

const vec2 vertices[3] = vec2[3](
	vec2(-1.0, -1.0),
	vec2( 3.0, -1.0),
	vec2(-1.0,  3.0)
);

out vec2 texcoord;

void main()
{
	vec2 vertex = vertices[gl_VertexID];
	gl_Position = vec4(vertex, 0.0, 1.0);

	texcoord = 0.5 * vertex + vec2(0.5);
}
)";

		char const separable_fragment[] =
R"(#version 330

uniform sampler2D u_input_texture;
uniform vec2 u_direction;

layout (location = 0) out vec4 out_color;

in vec2 texcoord;

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

void main()
{
	vec4 sum = vec4(0.0);

	for (int i = 0; i < N; ++i)
	{
		vec2 tc = texcoord + u_direction * float(i - M);
		sum += coeffs[i] * texture(u_input_texture, tc);
	}

	out_color = sum;
}
)";

		struct separable_impl
			: scene
		{
			separable_impl();

			void on_resize(int width, int height) override;

			void present() override;

		private:
			util::clock<std::chrono::duration<float>, std::chrono::high_resolution_clock> clock_;

			gfx::framebuffer fbo_1_;
			gfx::texture_2d color_buffer_1_;
			gfx::renderbuffer depth_buffer_1_;

			gfx::framebuffer fbo_2_;
			gfx::texture_2d color_buffer_2_;

			gfx::program blur_program_{separable_vertex, separable_fragment};

			gfx::array vao_;

			gfx::painter painter_;

			gfx::query_pool queries_;

			util::moving_average<float> frame_time_{32};
			util::moving_average<float> blur_time_{32};
		};

		separable_impl::separable_impl()
		{
			color_buffer_1_.nearest_filter();
			color_buffer_1_.clamp();

			color_buffer_2_.nearest_filter();
			color_buffer_2_.clamp();
		}

		void separable_impl::on_resize(int width, int height)
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

		void separable_impl::present()
		{
			float const dt = clock_.restart().count();
			frame_time_.push(dt);

			fbo_1_.bind();
			scene::draw();

			{
				auto scope = queries_.begin(gl::TIME_ELAPSED, [this](GLint value){ blur_time_.push(value / 1e6f); });

				fbo_2_.bind();

				gl::Clear(gl::COLOR_BUFFER_BIT);
				gl::Disable(gl::DEPTH_TEST);

				blur_program_.bind();
				blur_program_["u_input_texture"] = 0;
				blur_program_["u_direction"] = geom::vector{1.f / width(), 0.f};
				color_buffer_1_.bind(0);
				vao_.bind();

				gl::DrawArrays(gl::TRIANGLES, 0, 3);

				gfx::framebuffer::null().bind();

				gl::Clear(gl::COLOR_BUFFER_BIT);

				color_buffer_2_.bind(0);
				blur_program_["u_direction"] = geom::vector{0.f, 1.f / height()};

				gl::DrawArrays(gl::TRIANGLES, 0, 3);
			}

			{
				gfx::painter::text_options opts;
				opts.scale = 2.f;
				opts.c = gfx::black;
				opts.x = gfx::painter::x_align::left;
				opts.y = gfx::painter::y_align::top;

				painter_.text({20.f, 20.f}, "Separable", opts);

				painter_.text({20.f, 40.f}, util::to_string("FPS: ", 1.f / frame_time_.average()), opts);

				if (blur_time_.count() > 0)
					painter_.text({20.f, 60.f}, util::to_string("Blur: ", blur_time_.average(), "ms"), opts);
			}

			painter_.render(geom::window_camera{width(), height()}.transform());

			queries_.poll();
		}

	}

	std::unique_ptr<scene> separable()
	{
		return std::make_unique<separable_impl>();
	}

}
