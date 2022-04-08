#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include "EventHandler.h"
#include "Window.h"
#include <Rendering/View.h>

class Application
{
private:
	Application(uint32_t width, uint32_t height, const char *windowName)
	{
		mWindow.Create(width, height, windowName);

		//mView = new View;
		static View view;
		mView = &view;
		mView->OnCreate(width, height);
		EventHandler::SetListener(mView);
		MainLoop();
	}
	Application(Application &app) {}
	Application &operator=(Application &app) { return *this; }
	~Application()
	{
		mView->OnDestroy();
		//delete mView;
	}

	void MainLoop()
	{
		mView->OnInitialize();
		while (mWindow.WindowIsOpen())
		{
			

			mView->OnUpdate();

			mWindow.RenderGUI();


			mWindow.SwapBuffers();
			mWindow.PollEvents();
		}
		mView->OnFinalize();
	}

public:
	static Application &RunApplication(uint32_t width, uint32_t height, const char *windowName)
	{
		static Application app(width, height, windowName);
		return app;
	}

private:
	Window mWindow;
	View* mView;
};

#endif