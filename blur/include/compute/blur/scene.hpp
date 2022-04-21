#pragma once

#include <psemek/app/scene.hpp>

#include <memory>

namespace compute
{

	using namespace psemek;

	struct scene
		: app::scene_base
	{
		scene();
		~scene();

		void on_resize(int width, int height) override;

		void on_key_down(SDL_Keycode key) override;

	protected:

		void draw();

	private:

		struct impl;

		std::unique_ptr<impl> pimpl_;

		void replace_with(std::unique_ptr<scene> new_scene);
	};

	std::unique_ptr<scene> default_scene();

}
