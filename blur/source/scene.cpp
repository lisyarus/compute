#include <compute/blur/scene.hpp>

#include <psemek/app/app.hpp>
#include <psemek/gfx/gl.hpp>

namespace compute
{

	std::unique_ptr<scene> naive();

	struct scene::impl
	{

	};

	scene::scene()
		: pimpl_(std::make_unique<impl>())
	{}

	scene::~scene() = default;

	void scene::on_key_down(SDL_Keycode key)
	{
		app::scene_base::on_key_down(key);

		if (key == SDLK_0)
		{
			replace_with(naive());
		}
	}

	void scene::update()
	{

	}

	void scene::draw()
	{
		gl::ClearColor(0.8f, 0.8f, 1.f, 0.f);
		gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
	}

	void scene::replace_with(std::unique_ptr<scene> new_scene)
	{
		auto self = parent()->pop_scene();
		parent()->push_scene(std::move(new_scene));
	}

	std::unique_ptr<scene> default_scene()
	{
		return naive();
	}

}
