import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

screen_res = (800, 400)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
cell_size =2.51
mc_cell_size=0.5
mc_grid_size=(160,80)
cell_recpr = 1.0 / cell_size

line_color=0xdc143c
line_radius=1.0


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) *s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

dragCoefficient=0.4
h = 1.1
mass = 1.0
dim = 2
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
num_particles_x = 60
num_particles = num_particles_x * 20
max_num_particles_per_cell =100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

#碰撞后处理速度需要的参数
restitutionCoefficient=0.8
frictionCoeffient=1000

#迭代范围：(迭代需要用的参数)
minIterations = 2
maxIterations =20
maxError = 0.01
#静止密度
rho0 = 1.0
#松弛系数
w=0.5



neighbor_radius = h*1.05
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi


forces = ti.Vector.field(dim, float)
a_ii=ti.field(float)
d_ii=ti.Vector.field(dim,float)
add_d_ij_pj=ti.Vector.field(dim,float)
density_adv=ti.field(float)
pressure=ti.field(float)
density=ti.field(float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(positions, velocities,forces,d_ii,add_d_ij_pj)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)


nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

ti.root.dense(ti.i,num_particles).place(density,pressure,density_adv,a_ii)
ti.root.place(board_states)



#-----以下是Marching Cube需要用到的参数：
#-----坐标跟物体一样，网格数（160，80） 网格大小1.0
num_case=16
num_lines=10000
mc_radius=0.2
mc_squR=mc_radius**2

linePoints= ti.Vector.field(dim, float)
mcGridValue=ti.field(float)
newPos=ti.Vector.field(dim,float)
mcGridPos=ti.Vector.field(dim, float)

#存放每次计算的体素正方形的情况
mc_result=ti.field(int)
rea_line=ti.field(int)
ti.root.dense(ti.i, 2).place(newPos)
trangle_table=ti.Vector.field(4,ti.i32)

ti.root.dense(ti.i, num_case).place(trangle_table)
ti.root.dense(ti.i,num_lines).place(linePoints)
ti.root.place(rea_line)
ti.root.dense(ti.ij,mc_grid_size).place(mcGridPos)
ti.root.dense(ti.ij,mc_grid_size).place(mc_result)
ti.root.dense(ti.ij,mc_grid_size).place(mcGridValue)


#对应边，比如[2,3,-1,-1]对应CD边上一点和AC边上一点 连成的线段
#对应16中情况

trangle_table[0] = ti.Vector([-1, -1, -1, -1])
trangle_table[1] = ti.Vector([2,  3, -1, -1])
trangle_table[2] = ti.Vector([ 1,  2, -1, -1])
trangle_table[3] = ti.Vector([ 1,  3, -1, -1])
trangle_table[4] = ti.Vector([ 0,  1, -1, -1])
trangle_table[5] = ti.Vector([ 0,  3,  1,  2])
trangle_table[6] = ti.Vector([ 0,  2, -1, -1])
trangle_table[7] = ti.Vector([ 0,  3, -1, -1])
trangle_table[8] = ti.Vector([ 0,  3, -1, -1])
trangle_table[9] = ti.Vector([ 0,  2, -1, -1])
trangle_table[10] = ti.Vector([ 0,  1,  2,  3])
trangle_table[11] = ti.Vector([ 0,  1, -1, -1])
trangle_table[12] = ti.Vector([ 1,  3, -1, -1])
trangle_table[13] = ti.Vector([ 1,  2, -1, -1])
trangle_table[14] = ti.Vector([ 2,  3, -1, -1])
trangle_table[15] = ti.Vector([-1, -1, -1, -1])

#-----------------以下是marching cube需要用到的方法-----------------
#初始化网格点的位置，这步只进行一次
@ti.kernel
def init_mcGrid():
    for I in ti.grouped(mcGridPos):
        mcGridPos[I]=I*mc_cell_size


@ti.kernel
def resetmgGridValue():
   for I in ti.grouped(mcGridValue):
        #网格点势能的初始值为0
        mcGridValue[I] = 0.0

@ti.kernel
def calculatePointValue():
    for I in ti.grouped(mcGridPos):
        E=0.0
        x=mcGridPos[I]
        for k in range(num_particles):
            pos_k=positions[k]
            sqr_distance=(x-pos_k).norm_sqr()
            E += mc_squR/sqr_distance
        mcGridValue[I]=E

@ti.kernel
def getVoxelConfig():
    #遍历所有的网格点，第一行，最后一列由于无法构成体素正方形，所以除外：
    for i,j in ti.ndrange((0, mc_grid_size[0]-1), (1, mc_grid_size[1])):
        valueA = mcGridValue[i,j]
        valueB = mcGridValue[i+1, j]
        valueD = mcGridValue[i+1,j-1]
        valueC = mcGridValue[i,j-1]
        if valueA > 1.0:
            valueA = 1.0
        else:
            valueA = 0.0

        if valueB > 1.0:
            valueB = 1.0
        else:
            valueB = 0.0

        if valueD > 1.0:
            valueD = 1.0
        else:
            valueD = 0.0

        if valueC > 1.0:
            valueC = 1.0
        else:
            valueC = 0.0
        mc_result[i, j] = valueC * 1 + valueD * 2 + valueB * 4 + valueA * 8

        # if 0<=value<=15:
        #     mc_result[i, j]=value
        # #若=-1则表示mc_result计算错误，调试使用
        # else:
        #     mc_result[i, j]=-1

@ti.func
def sample(i0,j0,i1,j1):
    #线性插值部分，近似
    k=(1.0-mcGridValue[i0,j0]) / (mcGridValue[i1,j1]-mcGridValue[i0,j0])
    pos0 = mcGridPos[i0, j0]
    pos1 = mcGridPos[i1, j1]
    x=0.0
    y=0.0
    # 判断输入的边是水平边还是垂直边
    # 水平边,x相同
    if i0 == i1:
        x=pos0[0]
        y=pos0[1] + (pos1[1] - pos0[1]) * k
    #垂直边，y相同
    if j0==j1:
        x=pos0[0] + (pos1[0] - pos0[0]) * k
        y=pos0[1]
    return ti.Vector([x,y])

@ti.func
def newPos_clean():
    for I in range(2):
        newPos[I]=ti.Vector([0.0,0.0])

@ti.func
def linePoints_clean():
    for I in linePoints:
        linePoints[I]=ti.Vector([0.0,0.0])

@ti.kernel
def calculateLinePoint():
    index = 0
    # linePoints_clean()
    for i, j in ti.ndrange((0, mc_grid_size[0]-1), (1, mc_grid_size[1])):
        mc_case=mc_result[i,j]
        newPos_clean()
        if trangle_table[mc_case][0] != -1:
            for k in ti.static(range(2)):
                if trangle_table[mc_case][k] == 0:
                    newPos[k] = sample(i, j, i + 1, j)
                if trangle_table[mc_case][k] == 1:
                    newPos[k] = sample(i + 1, j, i + 1, j - 1)
                if trangle_table[mc_case][k] == 2:
                    newPos[k] = sample(i + 1, j - 1, i, j - 1)
                if trangle_table[mc_case][k] == 3:
                    newPos[k] = sample(i, j - 1, i, j)
            linePoints[index]=newPos[0]
            linePoints[index+1]=newPos[1]
            index+=2
        if trangle_table[mc_case][2] != -1:
            for k in ti.static(range(2,4)):
                if trangle_table[mc_case][k] == 0:
                    newPos[k-2] = sample(i, j, i + 1, j)
                if trangle_table[mc_case][k] == 1:
                    newPos[k-2] = sample(i + 1, j, i + 1, j - 1)
                if trangle_table[mc_case][k] == 2:
                    newPos[k-2] = sample(i + 1, j - 1, i, j - 1)
                if trangle_table[mc_case][k] == 3:
                    newPos[k-2] = sample(i, j - 1, i, j)
            linePoints[index] = newPos[0]
            linePoints[index + 1] = newPos[1]
            index +=2
    rea_line[None]=index

def run_marchingCubes():
    #重置节点的势能
    resetmgGridValue()
    #遍历每一个节点，计算势能
    calculatePointValue()
    #判断每一个网格点，若大于1则设为1，小于1设为0，并判断为16中情况的中的那一种
    getVoxelConfig()
    #根据每种情况，得出两个点位于那两条边上
    calculateLinePoint()

#-------------以上是Marching Cube 的操作----------------------------------------------------------


#----------------以下是iipsh操作---------------------------------------------



@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]

@ti.func
def confine_velocity_to_boundary(p,v):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world

    # 边界法向量（4个）
    norm_left = ti.Vector([1.0, 0.0])
    norm_down = ti.Vector([0.0, 1.0])
    norm_right = ti.Vector([-1.0, 0])
    norm_top = ti.Vector([0.0, -1.0])

    # 左边界
    if p[0] <= bmin:
        # 速度处理
        normalDotRelativeVel = norm_left.dot(v)
        relativeVelN = normalDotRelativeVel * norm_left
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0-frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    # 下边界
    if p[1] <= bmin:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_down
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0-frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    #右边界
    if p[0] >= bmax[0]:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_right
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0-frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    # 上边界
    if p[1] >= bmax[1]:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_top
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0-frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    return v

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p



@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])



def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    mc_linePos = linePoints.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
        mc_linePos[:,j]*= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect((0, 0), (board_states[None][0] / boundary[0], 1),
             radius=1.5,
             color=boundary_color)
    i=0
    while i < rea_line[None]:
        if mc_linePos[i,0]!=0.0 and mc_linePos[i,1]!=0.0:
            gui.line(begin=mc_linePos[i],end=mc_linePos[i+1],color=line_color,radius=line_radius)
        i+=2
    gui.show()

@ti.kernel
def prologue():
    # 清除网格
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    # 更新网格，计算每个粒子的归属网格
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    #建立每个粒子的邻居的搜索
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def advenction():
    #遍历所有的粒子,更新密度 计算d_ii，预测速度；
    for p_i in range(num_particles):
        pos_i = positions[p_i]
        temp_density = 0.0
        temp_add_kernel=ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            temp_density += mass * poly6_value(pos_ji.norm(), h)
            temp_add_kernel+=spiky_gradient(pos_ji, h)
        if temp_density > 0:
            density[p_i] = temp_density
        #如果没有邻居，暂时按0.1计算！！！！！
        else:
            density[p_i] = 0.1
        #因为0不能做除数，所以保证密度不能为0
        d_ii[p_i]=-time_delta**2*mass/density[p_i]**2*temp_add_kernel
        #计算预测速度,预测只考虑重力，阻力
        # 首先清除力
        forces[p_i] = ti.Vector([0.0, 0.0])
        # 加入重力
        forces[p_i] += ti.Vector([0.0, -9.8])
        # #加入阻力
        var = velocities[p_i]
        temp_force = var * (-dragCoefficient)
        forces[p_i] += temp_force
        #计算预测速度
        velocities[p_i]+=time_delta*(forces[p_i]/mass)

    #计算预测密度，初始压强，a_ii
    for p_i in range(num_particles):
        #初始压强
        pressure[p_i]=0.5*pressure[p_i]
        pos_i=positions[p_i]
        veli=velocities[p_i]
        temp_density_adv=0.0
        temp_aii=0.0
        for j in range(particle_num_neighbors[p_i]):
            # p_j 当前粒子的邻居
            p_j = particle_neighbors[p_i, j]
            velj=velocities[p_j]
            pos_ji = pos_i - positions[p_j]
            pos_ij=positions[p_j]-pos_i
            grad_ij=spiky_gradient(pos_ji, h)
            grad_ji=spiky_gradient(pos_ij, h)
            temp_density_adv+=time_delta*mass*(veli-velj).dot(grad_ij)
            dji=-time_delta**2*mass/density[p_i]**2*grad_ji
            temp_aii+=mass*(d_ii[p_i]-dji).dot(grad_ij)
        density_adv[p_i]=density[p_i]+temp_density_adv
        #避免出现分母为0的情况;
        if temp_aii==0:
            temp_aii=-0.01
           # print("cuole")
        #print(temp_aii)
        a_ii[p_i]=temp_aii


@ti.kernel
def Pressure_Solve():
    l=0
    eta= maxError * 0.01 * rho0
    chk = False
    while l<minIterations or (l<maxIterations and ~chk):
        #计算add_d_ij_pj
        chk=True
        for p_i in range(num_particles):
            temp_dij_pj=ti.Vector([0.0, 0.0])
            pos_i=positions[p_i]
            for j in range(particle_num_neighbors[p_i]):
                p_j = particle_neighbors[p_i, j]
                pressurej=pressure[p_j]
                posji=pos_i-positions[p_j]
                temp_dij_pj+=-time_delta**2*mass/density[p_j]**2*pressurej*spiky_gradient(posji,h)
            add_d_ij_pj[p_i]=temp_dij_pj
        density_avg = 0.0
        for p_i in range(num_particles):
            pos_i = positions[p_i]
            apart_pressure=0.0
            apart_density=0.0
            for j in range(particle_num_neighbors[p_i]):
                p_j = particle_neighbors[p_i, j]
                pressurej = pressure[p_j]
                posji=pos_i-positions[p_j]
                posij =positions[p_j]-pos_i
                d_ji=-time_delta**2*mass/density[p_i]**2*spiky_gradient(posij,h)
                k_notequal_i=add_d_ij_pj[p_j]-d_ji*pressure[p_i]
                apart_pressure+=mass*((add_d_ij_pj[p_i]-d_ii[p_j]*pressurej-k_notequal_i).dot(spiky_gradient(posji,h)))
                apart_density+=mass*((d_ii[p_i]*pressure[p_i]+add_d_ij_pj[p_i]-d_ii[p_j]*pressurej-k_notequal_i).dot(spiky_gradient(posji,h)))
            pressure[p_i]=(1.0-w)*pressure[p_i]+w/a_ii[p_i]*(rho0-density_adv[p_i]-apart_pressure)

            if pressure[p_i]<0:
                pressure[p_i]=0
            if pressure[p_i] == 0.0:
                density_avg+=rho0
            else :
                density_avg+=apart_density+density_adv[p_i]
        density_avg/=num_particles
        # print( density_avg)
        chk=chk and (density_avg-rho0<=eta)
        l+=1


@ti.kernel
def epilogue():
    # 计算压力
    for p_i in range(num_particles):
        den_pi = density[p_i]
        pressure_pi = pressure[p_i]
        pos_i = positions[p_i]
        pressureForces = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji =pos_i- positions[p_j]
            grad_ji = spiky_gradient(pos_ji, h)
            pressure_pj = pressure[p_j]
            den_pj = density[p_j]
            pressureForces -= mass * mass * (
                        pressure_pi / (den_pi * den_pi) + pressure_pj / (den_pj * den_pj)) * grad_ji
        #此时力用来修正速度，只有压力作用。所以可以直接在force里只放压力
        forces[p_i] = pressureForces
    # 这步更新速度位置
    for i in range(num_particles):
        a = forces[i] / mass
        velocities[i] += a * time_delta
        positions[i] += velocities[i] * time_delta
        velocities[i]=confine_velocity_to_boundary(positions[i],velocities[i])
        positions[i] = confine_position_to_boundary(positions[i])


def run_iisph():
    prologue()
    advenction()
    Pressure_Solve()
    epilogue()



def main():
    init_particles()
    init_mcGrid()
    gui = ti.GUI('MC', screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        run_iisph()
        run_marchingCubes()
        render(gui)


if __name__ == '__main__':
    main()
