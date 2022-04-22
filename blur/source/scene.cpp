#include <compute/blur/scene.hpp>

#include <psemek/app/app.hpp>
#include <psemek/gfx/gl.hpp>
#include <psemek/gfx/mesh.hpp>
#include <psemek/gfx/program.hpp>
#include <psemek/geom/camera.hpp>
#include <psemek/geom/rotation.hpp>
#include <psemek/geom/translation.hpp>
#include <psemek/geom/scale.hpp>
#include <psemek/cg/body/box.hpp>
#include <psemek/cg/body/icosahedron.hpp>
#include <psemek/util/clock.hpp>
#include <psemek/random/generator.hpp>
#include <psemek/random/uniform.hpp>
#include <psemek/random/uniform_sphere.hpp>

namespace compute
{

	std::unique_ptr<scene> naive();
	std::unique_ptr<scene> separable();
	std::unique_ptr<scene> separable_linear();
	std::unique_ptr<scene> compute();
	std::unique_ptr<scene> compute_lds();
	std::unique_ptr<scene> compute_separable();
	std::unique_ptr<scene> compute_separable_lds();
	std::unique_ptr<scene> compute_separable_single_lds();

	static char const simple_vertex[] =
R"(#version 330

uniform mat4 u_camera_transform;
uniform mat4 u_object_transform;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

out vec3 position;
out vec3 normal;

void main()
{
	position = (u_object_transform * vec4(in_position, 1.0)).xyz;
	gl_Position = u_camera_transform * vec4(position, 1.0);

	normal = (u_object_transform * vec4(in_normal, 0.0)).xyz;
}
)";

	static char const simple_fragment[] =
R"(#version 330

uniform vec4 u_object_color;
uniform vec3 u_ambient_light;
uniform vec3 u_light_direction;
uniform vec3 u_camera_position;

layout (location = 0) out vec4 out_color;

in vec3 position;
in vec3 normal;

void main()
{
	vec3 n = normalize(normal);

	float lit = max(0.0, dot(n, u_light_direction));

	vec3 camera_ray = normalize(u_camera_position - position);
	vec3 reflected = 2.0 * n * dot(n, u_light_direction) - u_light_direction;

	float specular = pow(max(0.0, dot(camera_ray, reflected)), 64.0);

	vec3 color = u_object_color.rgb * u_ambient_light + u_object_color.rgb * lit + vec3(specular);

	out_color = vec4(color, u_object_color.a);
}
)";

	struct scene::impl
	{
		util::clock<std::chrono::duration<float>, std::chrono::high_resolution_clock> clock;
		bool paused = false;
		float time = 0.f;

		gfx::program simple_program{simple_vertex, simple_fragment};
		gfx::mesh cube_mesh;

		geom::free_camera camera;

		struct cube_data
		{
			geom::point<float, 3> position;
			float size;
			geom::vector<float, 3> rotation_axis;
			float rotation_speed;
			geom::vector<float, 4> color;
		};

		std::vector<cube_data> cubes;

		impl()
		{
			cg::icosahedron<float> cube_body{{0.f, 0.f, 0.f}, 1.f};

			auto const & vertices = cg::vertices(cube_body);
			auto const & triangles = cg::triangles(cube_body);

			struct vertex
			{
				geom::point<float, 3> position;
				geom::vector<float, 3> normal;
			};

			std::vector<vertex> mesh_vertices;

			for (auto const & triangle : triangles)
			{
				auto v0 = vertices[triangle[0]];
				auto v1 = vertices[triangle[1]];
				auto v2 = vertices[triangle[2]];

				auto n = geom::normal(v0, v1, v2);

				mesh_vertices.emplace_back(v0, n);
				mesh_vertices.emplace_back(v1, n);
				mesh_vertices.emplace_back(v2, n);
			}

			cube_mesh.setup<geom::point<float, 3>, geom::vector<float, 3>>();
			cube_mesh.load(mesh_vertices, gl::TRIANGLES, gl::STATIC_DRAW);

			camera.near_clip = 0.1f;
			camera.far_clip = 100.f;
			camera.fov_y = geom::rad(60.f);
			camera.fov_x = camera.fov_y;

			camera.pos = {0.f, 0.f, 5.f};
			camera.axes[0] = {1.f, 0.f, 0.f};
			camera.axes[1] = {0.f, 1.f, 0.f};
			camera.axes[2] = {0.f, 0.f, 1.f};

			random::generator rng;
			random::uniform_sphere_vector_distribution<float, 3> random_unit_vector;

			int const count = 5;

			cubes.resize(count * count);

			for (int i = 0; i < cubes.size(); ++i)
			{
				auto & cube = cubes[i];

				cube.position = {(i % count) - (count - 1) / 2.f, (i / count) - (count - 1) / 2.f, 0.f};
				cube.size = 0.5f;
				cube.rotation_axis = random_unit_vector(rng);
				cube.rotation_speed = random::uniform<float>(rng, 0.25f, 0.5f);
				cube.color = {random::uniform<float>(rng), random::uniform<float>(rng), random::uniform<float>(rng), 1.f};
			}
		}

		static std::shared_ptr<impl> instance()
		{
			static std::weak_ptr<impl> weak_instance;

			if (auto ptr = weak_instance.lock())
				return ptr;

			auto ptr = std::make_shared<impl>();
			weak_instance = ptr;
			return ptr;
		}
	};

	scene::scene()
		: pimpl_(impl::instance())
	{}

	scene::~scene() = default;

	void scene::on_resize(int width, int height)
	{
		app::scene_base::on_resize(width, height);

		pimpl_->camera.set_fov(pimpl_->camera.fov_y, (width * 1.f) / height);
	}

	void scene::on_key_down(SDL_Keycode key)
	{
		app::scene_base::on_key_down(key);

		if (key == SDLK_1)
		{
			replace_with(naive());
		}
		else if (key == SDLK_2)
		{
			replace_with(separable());
		}
		else if (key == SDLK_3)
		{
			replace_with(separable_linear());
		}
		else if (key == SDLK_4)
		{
			replace_with(compute());
		}
		else if (key == SDLK_5)
		{
			replace_with(compute_lds());
		}
		else if (key == SDLK_6)
		{
			replace_with(compute_separable());
		}
		else if (key == SDLK_7)
		{
			replace_with(compute_separable_lds());
		}
		else if (key == SDLK_8)
		{
			replace_with(compute_separable_single_lds());
		}

		if (key == SDLK_SPACE)
		{
			pimpl_->paused = !pimpl_->paused;
		}
	}

	void scene::draw()
	{
		gl::Viewport(0, 0, width(), height());

		float const dt = pimpl_->clock.restart().count();

		if (!pimpl_->paused)
			pimpl_->time += dt;

		gl::ClearColor(0.8f, 0.8f, 1.f, 0.f);
		gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

		gl::Enable(gl::DEPTH_TEST);
		gl::DepthFunc(gl::LEQUAL);

		gl::Disable(gl::BLEND);
		gl::Enable(gl::CULL_FACE);

		pimpl_->simple_program.bind();
		pimpl_->simple_program["u_camera_transform"] = pimpl_->camera.transform();
		pimpl_->simple_program["u_camera_position"] = pimpl_->camera.position();
		pimpl_->simple_program["u_ambient_light"] = geom::vector{0.2f, 0.2f, 0.2f};
		pimpl_->simple_program["u_light_direction"] = geom::normalized(geom::vector{1.f, 1.f, 1.f});

		for (auto const & cube : pimpl_->cubes)
		{
			pimpl_->simple_program["u_object_transform"] =
				geom::translation<float, 3>(cube.position - cube.position.zero()).homogeneous_matrix() *
				geom::axis_rotation<float>(cube.rotation_axis, pimpl_->time * cube.rotation_speed).homogeneous_matrix() *
				geom::scale<float, 3>(cube.size).homogeneous_matrix();
			pimpl_->simple_program["u_object_color"] = cube.color;

			pimpl_->cube_mesh.draw();
		}
	}

	void scene::replace_with(std::unique_ptr<scene> new_scene)
	{
		auto app = parent();
		auto self = app->pop_scene();
		app->push_scene(std::move(new_scene));
	}

	std::unique_ptr<scene> default_scene()
	{
		return naive();
	}

}
