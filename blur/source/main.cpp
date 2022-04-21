#include <psemek/app/app.hpp>
#include <psemek/app/main.hpp>

#include <compute/blur/scene.hpp>

namespace compute
{

	using namespace psemek;

	struct blur_app
		: app::app
	{
		blur_app()
			: app::app("Blur", 0)
		{
			push_scene(default_scene());
		}
	};

}

int main()
{
	return psemek::app::main<compute::blur_app>();
}
