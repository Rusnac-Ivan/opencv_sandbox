#ifndef _GL_TEXTURE_

#include <Core/Platform.h>
#include <cassert>

namespace gl
{

	enum struct Target
	{
		TEXTURE_1D = GL_TEXTURE_1D,
		TEXTURE_2D = GL_TEXTURE_2D,
		TEXTURE_3D = GL_TEXTURE_3D,
		TEXTURE_1D_ARRAY = GL_TEXTURE_1D_ARRAY,
		TEXTURE_2D_ARRAY = GL_TEXTURE_2D_ARRAY,
		TEXTURE_RECTANGLE = GL_TEXTURE_RECTANGLE,

		TEXTURE_CUBE_MAP = GL_TEXTURE_CUBE_MAP,
		TEXTURE_CUBE_MAP_ARRAY = GL_TEXTURE_CUBE_MAP_ARRAY,
		TEXTURE_BUFFER = GL_TEXTURE_BUFFER,
		TEXTURE_2D_MULTISAMPLE = GL_TEXTURE_2D_MULTISAMPLE,
		TEXTURE_2D_MULTISAMPLE_ARRAY = GL_TEXTURE_2D_MULTISAMPLE_ARRAY
	};



	template<Target target>
	class Texture
	{
        enum Format
        {
            R       = GL_RED,
            RG      = GL_RG,
            RG,
            RG16,
            R16F,
            R32F,
            RG16F,
            RG32F,
            RGB,
            RGBA,
            RGB16,
            RGB16F,
            RGBA16,
            RGBA16F,
            RGB32F,
            RGBA32F,
            DEPTH,
            DEPTH32F
        };
	private:
		unsigned int mID;
		unsigned int mWidth;
		unsigned int mHeight;
	public:
		Texture();
		~Texture();

		void Bind();
		void UnBind();

		void Activate(const unsigned short& unit);
		void Deactivate(const unsigned short& unit);

		void 

	private:


	};

	template <Target target>
	Texture<target>::Texture() : mID(0), mWidth(0), mHeight(0)
	{
		glGenTextures(1, &mID);
		assert(mID != 0 && "Failed to generate texture!");
	}

	template <Target target>
	Texture<target>::~Texture()
	{
		if (mID != 0)
			glDeleteTextures(1, &mID);
	}



}


#endif // !_GL_TEXTURE_
