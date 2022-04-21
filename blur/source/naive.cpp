#include <compute/blur/scene.hpp>

#include <psemek/gfx/array.hpp>
#include <psemek/gfx/program.hpp>
#include <psemek/gfx/framebuffer.hpp>
#include <psemek/gfx/texture.hpp>
#include <psemek/gfx/renderbuffer.hpp>

namespace compute
{

	namespace
	{

		char const naive_vertex[] =
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

		char const naive_fragment[] =
R"(#version 330

uniform sampler2D u_input_texture;
uniform vec2 u_texture_size_inv;

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
		for (int j = 0; j < N; ++j)
		{
			vec2 tc = texcoord + u_texture_size_inv * vec2(float(i - M), float(j - M));
			sum += coeffs[i] * coeffs[j] * texture(u_input_texture, tc);
		}
	}

	out_color = sum;
}
)";

		struct naive_impl
			: scene
		{
			naive_impl();

			void on_resize(int width, int height) override;

			void present() override;

		private:
			gfx::framebuffer fbo_;
			gfx::texture_2d color_buffer_;
			gfx::renderbuffer depth_buffer_;

			gfx::program blur_program_{naive_vertex, naive_fragment};

			gfx::array vao_;
		};

		naive_impl::naive_impl()
		{
			color_buffer_.nearest_filter();
			color_buffer_.clamp();
		}

		void naive_impl::on_resize(int width, int height)
		{
			scene::on_resize(width, height);

			color_buffer_.load<gfx::color_rgba>({width, height});
			depth_buffer_.storage<gfx::depth24_pixel>({width, height});

			fbo_.color(color_buffer_);
			fbo_.depth(depth_buffer_);

			fbo_.assert_complete();
		}

		void naive_impl::present()
		{
			fbo_.bind();
			scene::draw();

			gfx::framebuffer::null().bind();

			gl::Clear(gl::COLOR_BUFFER_BIT);
			gl::Disable(gl::DEPTH_TEST);

			blur_program_.bind();
			blur_program_["u_input_texture"] = 0;
			blur_program_["u_texture_size_inv"] = geom::vector{1.f / width(), 1.f / height()};
			color_buffer_.bind(0);
			vao_.bind();
			gl::DrawArrays(gl::TRIANGLES, 0, 3);
		}

	}

	std::unique_ptr<scene> naive()
	{
		return std::make_unique<naive_impl>();
	}

}
