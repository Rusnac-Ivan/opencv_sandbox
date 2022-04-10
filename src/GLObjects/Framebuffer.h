#ifndef _FRAME_BUFFER_H_
#define _FRAME_BUFFER_H_

#include <Core/Platform.h>
#include <cassert>
#include <GLObjects/Texture.h>
#include <GLObjects/Vertices.h>
#include <GLObjects/Program.h>
#include <vector>
#include <memory>
#include <map>

namespace gl
{
	enum struct BindType
	{
		ReadAndDraw = GL_FRAMEBUFFER,
		Draw = GL_DRAW_FRAMEBUFFER,
		Read = GL_READ_FRAMEBUFFER,
	};

	enum struct AttachType
	{
		COLOR0 = GL_COLOR_ATTACHMENT0,
		COLOR1 = GL_COLOR_ATTACHMENT1,
		COLOR2 = GL_COLOR_ATTACHMENT2,
		COLOR3 = GL_COLOR_ATTACHMENT3,
		COLOR4 = GL_COLOR_ATTACHMENT4,
		COLOR5 = GL_COLOR_ATTACHMENT5,
		COLOR6 = GL_COLOR_ATTACHMENT6,
		COLOR7 = GL_COLOR_ATTACHMENT7,
		COLOR8 = GL_COLOR_ATTACHMENT8,
		COLOR9 = GL_COLOR_ATTACHMENT9,
		COLOR10 = GL_COLOR_ATTACHMENT10,
		COLOR11 = GL_COLOR_ATTACHMENT11,
		COLOR12 = GL_COLOR_ATTACHMENT12,
		COLOR13 = GL_COLOR_ATTACHMENT13,
		COLOR14 = GL_COLOR_ATTACHMENT14,
		COLOR15 = GL_COLOR_ATTACHMENT15,
		COLOR16 = GL_COLOR_ATTACHMENT16,

		DEPTH = GL_DEPTH_ATTACHMENT,
		STENCIL = GL_STENCIL_ATTACHMENT,
		DEPTH_STENCIL = GL_DEPTH_STENCIL_ATTACHMENT
	};


	struct Attachment
	{
		Texture2D texture;
		AttachType type;
		Format internal_format;
		Format format;
		DataType data_type;
	};

	class Framebuffer
	{

		unsigned int mID;

		gl::Vertices mQuad;
		Program* mProgram;

		uint32_t mWidth;
		uint32_t mHeight;

		int mPosAttribLoc;
		int mUVAttribLoc;

		std::vector<std::unique_ptr<Attachment>> mAttachments;
	public:
		Framebuffer();
		~Framebuffer();

		void Bind(BindType type);
		void UnBind(BindType type);

		void Init(Program* program, uint32_t width, uint32_t height);

		bool CheckFramebufferStatus();

		Texture2D* AttachTexture(AttachType attach_type, Format internal_format, Format format, DataType type, const std::map<ParamName, ParamValue>& param_val);

		void SetSizeBuffers(uint32_t width, uint32_t height);
		void DrawQuad();

		
	};

	Framebuffer::Framebuffer() : mWidth(0), mHeight(0), mProgram(nullptr), mPosAttribLoc(-1), mUVAttribLoc(-1)
	{
		glGenFramebuffers(1, &mID);
		assert(mID != 0 && "Failed to generate Framebuffer!");
	}

	Framebuffer::~Framebuffer()
	{
		glDeleteFramebuffers(1, &mID);
	}

	void Framebuffer::Bind(BindType type)
	{
		GLenum bind_type = static_cast<GLenum>(type);
		glBindFramebuffer(bind_type, mID);
	}

	void Framebuffer::UnBind(BindType type)
	{
		GLenum bind_type = static_cast<GLenum>(type);
		glBindFramebuffer(bind_type, NULL);
	}

	bool Framebuffer::CheckFramebufferStatus()
	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT)
		{
			assert("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");
		}
		else if (status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT)
		{
			assert("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT");
		}
		else if (status == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER)
		{
			assert("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER");
		}
		else if (status == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER)
		{
			assert("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER");
		}
		else if (status == GL_FRAMEBUFFER_UNSUPPORTED)
		{
			assert("GL_FRAMEBUFFER_UNSUPPORTED");
		}

		if (status == GL_FRAMEBUFFER_COMPLETE)
			return true;
		else
			return false;
	}

	void Framebuffer::Init(Program* program, uint32_t width, uint32_t height)
	{
		mProgram = program;

		if (mProgram)
		{
			//mPositionHandle = program->Attrib((ndsi8*)"a_position");
			//mTextureHandle = program->Attrib((ndsi8*)"a_texcoord0");
			mPosAttribLoc = mProgram->GetAttribLocation("a_Pos");
			mUVAttribLoc = mProgram->GetAttribLocation("a_UV");

			if (mPosAttribLoc == -1 || mUVAttribLoc == -1)
			{
				assert("Failed program for frame buffer quad.");
			}
		}

		const float fdim = 1.f;
		const int vert_count = 6;
		float quad_vert[] = {
			-fdim,  fdim, 0.f,  0.0f, 1.0f,
			-fdim, -fdim, 0.f,  0.0f, 0.0f,
			 fdim, -fdim, 0.f,  1.0f, 0.0f,

			-fdim,  fdim, 0.f,  0.0f, 1.0f,
			 fdim, -fdim, 0.f,  1.0f, 0.0f,
			 fdim,  fdim, 0.f,  1.0f, 1.0f
		};


		mQuad.AddVBO(std::vector<gl::AttribType>({ gl::AttribType::POSITION, gl::AttribType::TEXTURE_UV }), vert_count, sizeof(quad_vert), quad_vert);

		mWidth = width;
		mHeight = height;
	}

	Texture2D* Framebuffer::AttachTexture(AttachType attach_type, Format internal_format, Format format, DataType type, const std::map<ParamName, ParamValue>& param_val)
	{	
		std::unique_ptr<Attachment> attachment = std::make_unique<Attachment>();
		attachment->internal_format = internal_format;
		attachment->format = format;
		attachment->data_type = type;
		attachment->type = attach_type;
		attachment->texture.SetParameters(param_val);
		attachment->texture.SetTexture2D(0, internal_format, mWidth, mHeight, 0, format, type, nullptr);


		Bind(BindType::ReadAndDraw);
		GLenum texture_type = static_cast<GLenum>(TextureType::TARGET_2D);
		GLenum bind_type = static_cast<GLenum>(BindType::ReadAndDraw);
		glFramebufferTexture2D(bind_type, (GLenum)attach_type, texture_type, attachment->texture.GetId(), 0);
		UnBind(BindType::ReadAndDraw);

		Texture2D* ptr_texture = &attachment->texture;

		mAttachments.push_back(std::move(attachment));

		return ptr_texture;
	}


	


	void Framebuffer::SetSizeBuffers(uint32_t width, uint32_t height)
	{
		mWidth = width;
		mHeight = height;

		for (auto& attached : mAttachments)
		{
			attached->texture.SetTexture2D(0, attached->internal_format, mWidth, mHeight, 0, attached->format, attached->data_type, nullptr);
		}
	}

	void Framebuffer::DrawQuad()
	{
		if (mProgram)
		{
			mProgram->Use();
			mQuad.Draw(Primitive::TRIANGLES);
			mProgram->StopUsing();
		}
	}
}


#endif