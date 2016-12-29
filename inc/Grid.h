#ifndef _GRID_H
#define _GRID_H
#include "rsf.h"
void expand2d(float** vec2d_i, float** vec2d_o, sf_axis az, sf_axis ax, sf_axis azpad,
            sf_axis axpad);
void expand3d(float*** vec3d_i, float*** vec3d_o, sf_axis az, sf_axis ax, sf_axis ay,
              sf_axis azpad, sf_axis axpad, sf_axis aypad);
void window3d(float*** vec3d_o, float*** vec3d_i, sf_axis az, sf_axis ax, sf_axis ay,
              sf_axis azpad, sf_axis axpad, sf_axis aypad);
void wwin3d(float*** uo, float*** ui, int nzo, int nxo, int nyo, int nb);
void wwin2d(float** uo, float** ui, int nzo, int nxo, int nb);
#endif
