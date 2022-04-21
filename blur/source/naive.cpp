#include <compute/blur/scene.hpp>

namespace compute
{

	namespace
	{

		struct naive_impl
			: scene
		{

			void present() override;

		private:


		};

		void naive_impl::present()
		{
			scene::draw();
		}

	}

	std::unique_ptr<scene> naive()
	{
		return std::make_unique<naive_impl>();
	}

}
