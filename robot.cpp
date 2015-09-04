#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <iostream>
#include <ctype.h>
#include<windows.h>
using namespace cv;
using namespace std;

TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
Size subPixWinSize(10, 10), winSize(31, 31);

const int MAX_COUNT = 500;
bool needToInit = false;
bool nightMode = false;

int steps = 0;
int steps2 = 0;

int WINDOW_WIDTH = 600;
int WINDOW_HEIGHT = 600;

float angel1 = 0.0;
float angel2 = 0.0;
float angel3 = 0.0;

float motion = true;

/*light data*/
GLfloat light_position[] = { 0.0, 0.0, 10.0, 1.0 };  /* Infinite light location. */
GLfloat angle = 10;
//GLfloat light_direction[] = { .0, .0, -1.0 };

/*material data*/
GLfloat material_color1[] = { 0.1, .5, .0, 1.0 };
GLfloat material_color2[] = { .0, 1.0, 1.0, 1.0 };

/* Define data */
int current_x = 0, current_y = 0;
static GLdouble ex = .0, ey = .0, ez = 5.0, upx = .0, upy = 1.0, upz = .0, ox = .0, oy = .0, oz = 0.0;
int FX = 0, FY = 0;
/*Projection matrix mode*/
int projection = 0; 
Point2f point;
bool addRemovePt = false;

void initialize()
{
	/*initialize the window and light source*/
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glViewport(0, 0, (GLint)WINDOW_WIDTH, (GLint)WINDOW_HEIGHT);

	glLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, &angle);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

}
/*onMouse function for OpenCV*/
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	/*once click the mouse the current point will be recorded*/
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}
/*Ondisplay function for OpenGL*/
void onDisplay()
{
	/*open the camera*/
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		exit(-1);
	}

	namedWindow("robot", 1);
	setMouseCallback("robot", onMouse, 0);

	Mat gray, prevGray, image;
	vector<Point2f> points[2];

	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;
		/*catch one frame*/
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);
		/*make the whole vedio window to be dack*/
		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{
			/*automatic initialization*/
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;
		}
		/*manually catch the feature point*/
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			/*capture feature in every frames*/
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;
				/*add the point to the list*/
				points[1][k++] = points[1][i];

				/*compute the rotation distance*/
				FX = points[0][points[0].size() - 1].x - points[1][points[1].size() - 1].x;
				FY = points[0][points[0].size() - 1].y - points[1][points[1].size() - 1].y;
				/*detect the error point*/
		//		Sleep(1000);	
		//		printf("%f,%f\n", points[1][points[1].size() - 1].x, points[1][points[1].size() - 1].y);
				/*draw red point on the image*/
				circle(image, points[1][i], 3, Scalar(0, 0, 255), -1, 8);
			}
			/*reset the size*/
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
			points[1].push_back(tmp[0]);	
			addRemovePt = false;
		}
		needToInit = false;
		imshow("robot", image);
		
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);

		/*Draw robot image with OpenGL and show 3D*/
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		/*glMatrixMode(GL_PROJECTION);*/
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glOrtho(-3, 3, -3, 3, -100, 100);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		gluLookAt(ex, ey, ez, ox, oy, oz, upx, upy, upz);
		/*right position for the robot*/
		glTranslatef(0.0, 1.5, 0.0);

		glMaterialfv(GL_FRONT, GL_DIFFUSE, material_color1);
		GLUquadricObj *quadratic;
		quadratic = gluNewQuadric();

		glRotatef(angel3, 0, 1, 0);
		/*body*/
		glRotated(90, 1, 0, 0);
		glPushMatrix();
		gluCylinder(quadratic, 0.3, 0.3, 1.0, 32, 32);
		glPopMatrix();
		glRotated(-90, 1, 0, 0);

		/*between head and body*/
		glPushMatrix();
		glutSolidSphere(0.3, 32, 32);
		glPopMatrix();
		/*under body*/
		glTranslatef(0.0, -1.0, 0.0);
		glPushMatrix();
		glutSolidSphere(0.3, 32, 32);
		glPopMatrix();
		glTranslatef(.0, 1.0, .0);

		/*head*/
		glTranslatef(.0, .6, .0);
		glPushMatrix();
		glutSolidSphere(0.3, 32, 32);
		glPopMatrix();


		glTranslatef(0.0, -0.6, 0.0);
		/*left arm one*/
		glRotatef(angel1, 1, 0, 0);
		glTranslatef(-0.4, 0.0, 0.0);
		glPushMatrix();
		glutSolidSphere(0.1, 32, 32);
		glPopMatrix();
		/*left arm two*/
		glRotatef(90, 1, 0, 0);
		glRotatef(-45, 0, 1, 0);
		glPushMatrix();
		gluCylinder(quadratic, 0.1, 0.1, 0.5, 32, 32);
		glPopMatrix();
		glRotatef(45, 0, 1, 0);
		glRotatef(-90, 1, 0, 0);
		/*left hand*/
		glTranslatef(-0.35, -0.35, 0);
		glPushMatrix();
		glutSolidSphere(0.1, 32, 32);
		glPopMatrix();
		glTranslatef(0.35, 0.35, 0);
		glRotatef(-angel1, 1, 0, 0);

		glTranslatef(0.4, 0.0, 0.0);
		/*right arm one*/
		glRotatef(angel2, 1, 0, 0);
		glTranslatef(0.4, 0.0, 0.0);
		glPushMatrix();
		glutSolidSphere(0.1, 32, 32);
		glPopMatrix();
		/*right arm two*/
		glRotatef(90, 1, 0, 0);
		glRotatef(45, 0, 1, 0);
		glPushMatrix();
		gluCylinder(quadratic, 0.1, 0.1, 0.5, 32, 32);
		glPopMatrix();
		glRotatef(-45, 0, 1, 0);
		glRotatef(-90, 1, 0, 0);
		/*right hand*/
		glTranslatef(0.35, -0.35, 0);
		glPushMatrix();
		glutSolidSphere(0.1, 32, 32);
		glPopMatrix();
		glTranslatef(-0.35, 0.35, 0);
		glRotatef(-angel2, 1, 0, 0);
		glTranslatef(-0.4, 0.0, 0.0);

		/*legs*/
		glTranslatef(0, -1.0, 0);
		/*left leg*/
		glRotatef(angel2, 1, 0, 0);
		glTranslatef(-0.2, 0, 0);
		glRotatef(90, 1, 0, 0);
		glPushMatrix();
		gluCylinder(quadratic, 0.1, 0.1, 1.0, 32, 32);
		glPopMatrix();
		glRotatef(-90, 1, 0, 0);
		/*left foot*/
		glTranslatef(0, -1.0, 0);
		glutSolidSphere(0.1, 32, 32);
		glTranslatef(0, 1.0, 0);
		glTranslatef(0.2, 0, 0);
		glRotatef(-angel2, 1, 0, 0);

		glRotatef(angel1, 1, 0, 0);
		/*right leg*/
		glTranslatef(0.2, 0, 0);
		glRotatef(90, 1, 0, 0);
		glPushMatrix();
		
		gluCylinder(quadratic, 0.1, 0.1, 1.0, 32, 32);
		glPopMatrix();
		glRotatef(-90, 1, 0, 0);

		/*right foot*/
		glTranslatef(0, -1.0, 0);
		glutSolidSphere(0.1, 32, 32);
		glTranslatef(0, 1.0, 0);
		glTranslatef(-0.2, 0, 0);

		glRotatef(-angel1, 1, 0, 0);

		glTranslatef(0, 1.0, 0);
		/*head decloration*/
		glTranslatef(0, 0.6, 0);
		/*left*/
		glTranslatef(-0.3, 0, 0);
		glutSolidSphere(0.1, 32, 32);
		glTranslatef(0.3, 0, 0);
		/*right*/
		glTranslatef(0.3, 0, 0);
		glutSolidSphere(0.1, 32, 32);
		glTranslatef(-0.3, 0, 0);
		glTranslatef(0, -0.6, 0);

		/*eyes*/
		glMaterialfv(GL_FRONT, GL_DIFFUSE, material_color2);
		glTranslatef(0, 0.6, 0.3);
		/*left*/
		glTranslatef(-0.1, 0, 0);
		glutSolidSphere(0.05, 32, 32);
		glTranslatef(0.1, 0, 0);
		/*right*/
		glTranslatef(0.1, 0, 0);
		glutSolidSphere(0.05, 32, 32);
		glTranslatef(-0.1, 0, 0);
		glTranslatef(0, -0.6, -0.3);
		glRotatef(-angel3, 0, 1, 0);
		
		glFlush();
		glutSwapBuffers();//display the buffer

		/*make the robot walking*/
		if (steps == 0)
		{
			angel1 -= 5;
			if (angel1 == -45)
				steps = 1;
		}
		else if (steps == 1)
		{
			angel1 += 5;
			if (angel1 == 45)
				steps = 0;
		}
		if (steps2 == 0)
		{
			angel2 += 5;
			if (angel2 == 45)
				steps2 = 1;

		}
		else if (steps2 == 1)
		{
			angel2 -= 5;
			if (angel2 == -45)
				steps2 = 0;
		}
		/*rotate and swift the robot*/
		if ((abs(FX) - abs(FY)) > 0)
		{
			ex += static_cast<float>(FX) / 10.f;
			ey += static_cast<float>(FY) / 10.f;
		}
		else
		{	
			upx += static_cast<float>(FY) / 10.f;
			upy += static_cast<float>(FY) / 10.f;			
		}
		glutPostRedisplay();
		char c = (char)waitKey(10);
		if (c == 27)
		{
			cv::destroyAllWindows();
			break;
		}
		switch (c)
		{
		/*catch the outline*/
		case 'r':
			needToInit = true;
			break;
		/*clear the select point*/
		case 'c':
			points[0].clear();
			points[1].clear();
			break;
		/*make the camera closed*/
		case 'n':
			nightMode = !nightMode;
			break;
		}
	}
}
void onKeyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		glutDestroyWindow(1);
		exit(1);

		break;
	default:
		break;
	}
}

void onIdle()
{
	glutPostRedisplay();
}
int main(int argc, char** argv)
{

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(30, 60);

	glutCreateWindow("ROBOT");
	initialize();

	glutDisplayFunc(onDisplay);
	glutKeyboardFunc(onKeyboard);

	glutIdleFunc(onIdle);
	glutMainLoop();
	return 1;
}
