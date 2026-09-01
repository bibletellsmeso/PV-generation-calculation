import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys

np.set_printoptions(precision=6, suppress=True)

# https://bd.kma.go.kr/kma2020/fs/energySelect1.do?pageNum=5&menuCd=F050701000
Local = {'local_latitude' : 19.728144, 'local_longitude' : 156.058936, 'standard_longitude' : 150, 'year' : 2018, 'month' : 7, 'day' : 4, 'hour' : 12, 'min' : 0}
Angle = {'sun_k': 1355, 'elevation_angle': 30, 'panel_azimuth': 180, 'panel_tilted_angle': 30, 'zenith_angle': 60, 'rho': 0.2}
PV = {'alpha': -0.0035, 'gamma': 0.02, 'EFF_STC': 1, 'Power_Capacity': 500}


# 각도 단위 변경
d2r = math.pi / 180
r2d = 180 / math.pi

class Photovoltaic:
    def __init__(self):
        self.Sim_time = 96 # 15min
        self.Weather_Forecast = np.genfromtxt('Hawaii weather.csv', delimiter=',', skip_header=1, dtype=np.float32)
        self.real_PV = np.loadtxt("PV_for_scheduling.txt")
        self.GHI = self.Weather_Forecast[:,2]
        self.Tem = self.Weather_Forecast[:,3]
        # self.Predict_PV = self.Weather_Forecast[:,1]
        
        self.emtpy_Predict_PV = np.full(self.Sim_time, np.nan)
        self.new_GHI = np.zeros(self.Sim_time)
        self.new_Tem = np.zeros(self.Sim_time)
        
        for i in range(95):
            self.new_GHI[i+1] = self.GHI[15*i]
            self.new_Tem[i+1] = self.Tem[15*i]
        self.new_GHI[0] = self.new_GHI[1]
        self.new_Tem[0] = self.new_Tem[1]    
        self.new_GHI[-1] = self.new_GHI[-2]
        self.new_Tem[-1] = self.new_Tem[-2]
    def Local_Solar(self, Local):
        # 하와이
        self.local_latitude = Local['local_latitude']               # 위도
        self.local_longitude = Local['local_longitude']             # 경도
        self.standard_longitude = Local['standard_longitude']       # 자오선
        self.year = Local['year']
        self.month = Local['month']
        self.day = Local['day']
        self.hour = Local['hour']                                   # 정오 고정
        self.min = Local['min']

        self.x = np.array(range(24))                                # 시변
        self.hour = np.repeat(self.x, 4)           
        self.y = np.array([0, 15, 30, 45])
        self.min = np.tile(self.y, 24)

        # 균시차 (Equaiont of Time), 진태양시와 평균태양시의 차이
        self.day_of_year = np.array(datetime(self.year, self.month, self.day).timetuple().tm_yday)
        B = (self.day_of_year - 1) * 360/365
        EOT = 229.2 * (0.000075 + 0.001868 * np.cos(d2r * B) - 0.032077 * np.sin(d2r * B) - 0.014615 * np.cos(d2r * 2 * B) - 0.04089 * np.sin(d2r * 2 * B))

        # 시간각 (Hour Angle), 정남향일 때 0도 기준, 동쪽 [-], 서쪽 [+]
        self.local_hour_decimal = self.hour + self.min / 60
        self.delta_longitude = self.local_longitude - self.standard_longitude
        self.solar_time_decimal = (self.local_hour_decimal * 60 + 4 * self.delta_longitude + EOT) / 60
        self.solar_time_hour = np.asarray(self.solar_time_decimal, dtype = int)
        self.solar_time_min = (self.solar_time_decimal * 60) % 60
        self.hour_angle = (self.local_hour_decimal * 60 + 4 * self.delta_longitude + EOT) / 60 * 15 - 180

        # 태양 적위 (Solar Declination), 태양의 중심과 지구의 중심을 연결하는 선과 지구 적도면이 이루는 각
        if (3 < self.month < 10) or (self.month == 3 and self.day >= 21) or (self.month == 9 and self.day <= 22):
             self.ecliptic_obliquity = 23.45
        else:
            self.ecliptic_obliquity = -23.45
        self.solar_declination = self.ecliptic_obliquity * np.sin(d2r * 360 / 365 * (284 + self.day_of_year))

        # 태양 고도 (Solar Altitude), 태양과 지표면이 이루는 각
        self.solar_altitude = r2d * np.arcsin(np.cos(d2r * self.local_latitude) * np.cos(d2r * self.solar_declination) * np.cos(d2r * self.hour_angle) + np.sin(d2r * self.local_latitude) * np.sin(d2r * self.solar_declination))

        # 태양 천정각 (Solar Zenith)
        self.solar_zenith = r2d * np.arccos(np.sin(d2r * self.local_latitude) * np.sin(d2r * self.solar_declination) + np.cos(d2r * self.local_latitude) * np.cos(d2r * self.solar_declination) * np.cos(d2r * self.hour_angle))

        # 태양 방위각 (Solar Azimuth) - 남향을 기준으로 동쪽 [+], 서쪽 [-]
        self.solar_azimuth = r2d * np.arccos(np.sin(d2r * self.solar_altitude) * np.sin(d2r * self.local_latitude) - np.sin(d2r * self.solar_declination)) / (np.cos(d2r * self.solar_altitude) * np.cos(d2r * self.local_latitude))
        # print("Solar Azimuth is", solar_azimuth)

        # 태양 방위각2 - 북향을 기준으로
        self.solar_azimuth_2 = r2d * np.arccos((np.sin(d2r * self.solar_declination) * np.cos(d2r * self.local_latitude) - np.cos(d2r * self.hour_angle) * math.cos(d2r * self.solar_declination) * np.sin(d2r * self.local_latitude)) / np.sin(d2r * (90 - self.solar_altitude)))

        # 반사율
        self.cal_rho = 0.012 * self.solar_zenith - 0.04

    def POA(self, Angle):
        self.panel_azimuth = Angle['panel_azimuth']                         # 방위각, 정남형(180) -45~45
        self.panel_tilted_angle = Angle['panel_tilted_angle']               # 패널경사각-에기연, 제주 24
        # self.solar_zenith = 30                                              # 각도 고정
        # self.solar_altitude = 90 - self.solar_zenith
        # self.solar_azimuth_2 = 180
        # 경사면 입사각(The Angle of Incidence) 방법 1
        self.angle_of_incidence = r2d * np.arccos(np.cos(d2r * self.solar_zenith) * np.cos(d2r * self.panel_tilted_angle) + np.sin(d2r * self.solar_zenith) * np.sin(d2r * self.panel_tilted_angle) * np.cos(d2r * self.solar_azimuth_2 - d2r * self.panel_azimuth))
        
        # # 방법 2
        # self.angle_of_incidence = np.arccos(np.sin(d2r * solar_altitude) * np.cos(d2r * self.panel_tilted_angle)\
        #     + np.cos(d2r * self.solar_azimuth_2 - d2r * self.panel_tilted_angle) * np.cos(d2r * solar_altitude)\
        #         * np.sin(d2r * self.panel_tilted_angle)) * r2d

        # 법선면 직달일사량 (Direct Normal Irradiance)
        self.DNI = np.zeros(self.Sim_time)
        self.Io = Angle['sun_k']                                                       # 대기외 일사량, 태양 상수
        self.Ir_oH = self.Io * np.cos(d2r * self.solar_zenith)
        self.pp = self.new_GHI / self.Ir_oH

        for i in range(self.Sim_time):
            if 1540 * self.pp[i] - 470 > 860:
                self.DNI[i] = 860
            elif 0 < 1540 * self.pp[i] - 470 <= 860:
                self.DNI[i] = 1540 * self.pp[i] - 470
            else:
                self.DNI[i] = 0

        # 관측값 없을 때
        # self.DNI = 910 * np.sin(d2r * self.solar_altitude) + 0.25 * (910 * np.sin(2 * d2r * self.solar_altitude)) # 652.02

        # 경사면 직달일사량 (Plane of Array Beam component)
        self.POA_b = self.DNI * np.cos(d2r * self.angle_of_incidence)

        # 경사면 확산일사량 (POA Sky-Diffuse component)
        self.DHI = self.new_GHI - self.DNI * np.sin(d2r * self.solar_altitude)   # 수평면 확산일사량 (Diffuse Horizontal irradiance)
        self.POA_d = self.DHI * ((1 + np.cos(d2r * self.panel_tilted_angle))) / 2 + self.new_GHI * (0.12 * d2r * self.solar_zenith - 0.04) * (1 - np.cos(d2r * self.panel_tilted_angle)) / 2

        # self.POA_d = np.zeros(self.Sim_time)
        # for i in range(self.Sim_time):
        #     if self.new_GHI[i] >= self.POA_b[i] + np.sin(d2r * self.solar_altitude[i]):
        #         self.POA_d[i] = (self.new_GHI[i] - self.POA_b[i] - np.sin(d2r * self.solar_altitude[i])) * ((1 + np.cos(d2r * self.panel_tilted_angle))) / 2
        #     else:
        #         self.POA_d[i] = 0

        # 지표면 반사일사량 (Ground-Reflected component)
        self.rho = Angle['rho']
        self.POA_g = self.rho * self.new_GHI * ((1 - np.cos(d2r * self.panel_tilted_angle)) / 2)

        # 경사면 전일사량(Plane of Array Irradiance)
        self.Ir_POA = self.POA_b + self.POA_d + self.POA_g
        for i in range(self.Sim_time):
            if self.Ir_POA[i] >= 0:
                self.Ir_POA[i] = self.Ir_POA[i]
            else:
                self.Ir_POA[i] = 0

    def Module_Tem(self, PV):
        self.alpha = PV['alpha']   # \degc (crystalline Siicon (cSi)), -0.00415 (우리나라 모듈 출력온도계수 평균)
        self.gamma = PV['gamma']   # free-standing installations = 0.02 / roof integrated systems = 0.056
        self.T_m = self.new_Tem + self.gamma * self.Ir_POA

    def Normal_EFF(self):
        for i in range(self.Sim_time):
            if self.new_GHI[i] <= 50:
                self.EFF_norm[i] = 0.91
            elif 50 < self.new_GHI[i] <= 150:
                self.EFF_norm[i] = (0.95-0.91) / (150-50) * (self.new_GHI[i] - 50) + 0.91
            elif 150 < self.new_GHI[i] <= 250:
                self.EFF_norm[i] = (0.98-0.95) / (250-150) * (self.new_GHI[i] - 150) + 0.95
            elif 250 < self.new_GHI[i] <= 350:
                self.EFF_norm[i] = (0.99-0.98) / (350-250) * (self.new_GHI[i] - 250) + 0.98
            elif 350 < self.new_GHI[i] <= 450:
                self.EFF_norm[i] = (1.00-0.99) / (450-350) * (self.new_GHI[i] - 350) + 0.99 
            elif 450 < self.new_GHI[i] <= 550:
                self.EFF_norm[i] = (1.02-1.00) / (550-450) * (self.new_GHI[i] - 450) + 1.00
            elif 550 < self.new_GHI[i] <= 650:
                self.EFF_norm[i] = (1.03-1.02) / (650-550) * (self.new_GHI[i] - 550) + 1.02
            elif 650 < self.new_GHI[i] <= 750:
                self.EFF_norm[i] = (1.04-1.03) / (750-650) * (self.new_GHI[i] - 650) + 1.03
            elif 750 < self.new_GHI[i] <= 850:
                self.EFF_norm[i] = (1.03-1.04) / (850-750) * (self.new_GHI[i] - 750) + 1.04
            elif 850 < self.new_GHI[i] <= 950:
                self.EFF_norm[i] = (1.01-1.03) / (950-850) * (self.new_GHI[i] - 850) + 1.03
            elif 950 < self.new_GHI[i] <= 1050:
                self.EFF_norm[i] = (0.99-1.01) / (1050-950) * (self.new_GHI[i] - 950) + 1.01
            elif 1050 < self.new_GHI[i] <= 1150:
                self.EFF_norm[i] = (0.98-0.99) / (1150-1050) * (self.new_GHI[i] - 1050) + 0.99
            elif 1150 < self.new_GHI[i] <= 1250:
                self.EFF_norm[i] = (0.97-0.98) / (1250-1150) * (self.new_GHI[i] - 1150) + 0.98
            else:
                self.EFF_norm[i] = 0.97
        return self.EFF_norm

    def PV_Gen(self, PV):
        self.alpha = PV['alpha']
        self.P_nom =  PV['Power_Capacity']

        # self.a_1, self.a_2, self.a_3 = 1, 2, 3
        # self.EFF_25 = self.EFF_25 = self.a_1 + self.a_2 * self.new_GHI + self.a_3 * math.log10(self.new_GHI)
        # self.EFF_t = self.EFF_25 * (1 + self.alpha * (self.T_m - 25))
        # self.EFF_STC = PV['EFF_STC']        # 18 % (Polycrystalline silicon module)

        self.EFF_norm = np.zeros(self.Sim_time)
        self.EFF_norm = self.Normal_EFF()
        self.PV_output = self.EFF_norm * (1 + self.alpha * (self.T_m - 25)) * (self.Ir_POA / 1000) * self.P_nom
        for i in range(self.Sim_time):
            if self.PV_output[i] > 0:
                self.PV_output[i] = self.PV_output[i]
            else:
                self.PV_output[i] = 0
        self.PV_output_g = self.EFF_norm * (1 + self.alpha * (self.new_Tem - 25)) * (self.new_GHI / 1000) * self.P_nom


        self.error = (self.real_PV - self.PV_output)
        self.rmse = 1 / np.sqrt(self.Sim_time) * np.sqrt(np.sum(self.error**2))
        self.relative_error = (self.PV_output - self.real_PV) / self.real_PV * 100
        self.MAE = 1 / self.Sim_time * np.sum(np.abs(self.error))

        self.PV_stack = np.zeros(self.Sim_time)
        for i in range(24-1):
                self.PV_stack[4*(i+1)] = self.PV_stack[4*i] + self.PV_output[4*i] + self.PV_output[4*i+1] + self.PV_output[4*i+2] + self.PV_output[4*i+3]
        self.time_h = np.zeros(self.Sim_time)
        for i in range(self.Sim_time-1):
            self.time_h[0] = 0
            self.time_h[i+1] = self.time_h[i] + 1

        # plt.plot(self.PV_capacity, 'b-', self.PV_output, 'ro-,', self.T_m)

        plt.style.use(['science', 'no-latex'])
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False

        # # 고도 및 방위각
        # fig = plt.figure(1, figsize=(12,9))
        # plt.xlabel('Time step(15m)', fontsize=18)
        # plt.ylabel(r'Degree($^{\circ}$)', fontsize=18)
        # # plt.suptitle("Solar Altitude and Azimuth", fontsize = 14)
        # plt.plot(self.solar_altitude, '-', lw=2, color='mediumblue', label = 'Altitude')
        # plt.plot(self.solar_azimuth_2, '-', lw=2, color='crimson', label = 'Azimuth')
        # plt.tight_layout()
        # fig.legend(fontsize=18)

        # # 일사량 비교
        # fig = plt.figure(2, figsize=(12,9))
        # plt.xlabel('Time step(15m)', fontsize=18)
        # plt.ylabel(r'Irradiance(W/${m^2}$)', fontsize=18)
        # # plt.suptitle("Horizontal and Plane of array Irradiance", fontsize = 14)
        # plt.plot(self.new_GHI, '--', lw=2, marker=10, color='mediumblue', label = 'Ir_global', zorder=2)
        # plt.plot(self.Ir_POA, '-', lw=2, marker=11, color='crimson', label = 'Ir_local', zorder=1)
        # plt.tight_layout()
        # fig.legend(fontsize=18)

        self.PV_output_h = np.zeros(24)
        for i in range(24):
            self.PV_output_h[i] = self.PV_output[4*i]
        self.hour = list(range(24))
        self.PV_upper_bid = self.PV_output_h * 1.12

        self.kpx_PV_data = pd.read_csv('KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], dtype={'Date': str, 'Hour': str, 'Power': str}, encoding='CP949')[1:] # dtype = DataFrame
        self.kpx_PV_data = pd.DataFrame(self.kpx_PV_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)
        self.kpx_PV = []
        for i in range(24):
            self.kpx_PV.append(self.kpx_PV_data[self.kpx_PV_data[:, 0] == i, -1])

        self.PV_var = np.array([])
        for i in range(24):
            self.PV_var = np.append(self.PV_var, np.nanvar(self.kpx_PV[i] / np.max(self.kpx_PV)))

        self.PV_neg = self.PV_output_h * 1.96 * np.sqrt(self.PV_var) / np.sqrt(1)
        self.PV_lower_bid = self.PV_output_h - self.PV_neg

        fig = plt.figure(5, figsize=(12,9))
        plt.xlabel('Time step (h)', fontsize=18)        
        plt.ylabel('Power (kW)', fontsize=18)
        plt.bar(self.hour, self.PV_output_h, label='PV output local')
        plt.plot(self.hour, self.PV_output_h, marker='o', lw=2, label='Bid capacity')
        plt.plot(self.hour, self.PV_upper_bid, marker='^', lw=2, label='Risk-averse')
        plt.plot(self.hour, self.PV_lower_bid, marker='v', lw=2, label='Opportunity-seeking')
        fig.legend(fontsize=18)



        # # 온도 비교
        # fig = plt.figure(3, figsize=(12,9))
        # plt.xlabel('Time step(15m)', fontsize=18)
        # plt.ylabel(r'Temperature($^{\circ}$C)', fontsize=18)
        # # plt.suptitle("Temperature and Module Temperature", fontsize = 14)
        # plt.plot(self.new_Tem, '--', lw=2, color='mediumblue', label = 'T_global', zorder=2)
        # plt.plot(self.T_m, '-', lw=2, color='crimson', label = 'T_local', zorder=1)
        # plt.tight_layout()
        # fig.legend(fontsize=18)

        # fig = plt.figure(4, figsize=(12,9))
        # plt.xlabel('Time step(15m)', fontsize=18)
        # plt.ylabel('PV output(kW)', fontsize=18)
        # plt.plot(self.real_PV, '--', lw=2, color='mediumblue', label = 'Real')
        # plt.plot(self.PV_output, '-', lw=2, color='crimson', label = 'Predict')
        # plt.tight_layout()
        # fig.legend(fontsize=18)

        # # 일사량 및 온도와 발전량 관계
        # fig, ax1 = plt.subplots(1, figsize=(12,9))    # 직접 figure 객체를 생성
        # ax1.plot(self.PV_output , 'o-', lw=2, color='crimson', label='PV output', zorder=2)
        # ax1.plot(self.new_GHI , '-', lw=2, color='mediumblue', label='GHI', zorder=3)
        # ax1.set_xlabel('Time step(15m)', fontsize=18)
        # ax1.set_ylabel(r'GHI(W/${m^2}$) & PV output(kWh)', fontsize=18)
        # # plt.suptitle("PV Output Correlation", fontsize = 14)
        # # ax1.minorticks_on()
        # ax2 = ax1.twinx()
        # ax2.plot(self.new_Tem , '--', lw=2, color='#FA7268', label='Tem', zorder=1)
        # ax2.set_ylabel(r'Temperature($^{\circ}$C)', fontsize=18)       
        # ax1.patch.set_visible(False)
        # ax1.set_zorder(1)
        # lines = []
        # labels = []
        # for ax in fig.axes:
        #     axLine , axLabel = ax.get_legend_handles_labels()
        #     lines.extend(axLine)
        #     labels.extend(axLabel)
        
        # ax1.legend(lines, labels, loc="best", fontsize=18)
        
        # # 발전량 계산
        # fig, ax1 = plt.subplots(1, figsize=(12,9))
        # ax1.plot(self.PV_output, '-', marker='o', lw=2, color='crimson', label='PV output_local', zorder=2)
        # # ax1.plot(self.Predict_PV, '--', lw=2, color='forestgreen', alpha=0.7, label='Predcition', zorder=1)
        # # ax1.plot(self.PV_output_g, '-', lw=2, marker=10, color='mediumblue', label='PV output_global', zorder=0)
        # ax1.plot(self.T_m, '-.', lw=2, color='#FA7268', label='Module temperate', zorder=1)             
        # ax1.hlines(y=self.P_nom, xmin=0, xmax=96, color='blueviolet', linestyles='solid', label='Capacity', lw=2, zorder=1)
        # ax1.set_xlabel('Time step(15m)', fontsize=18)
        # ax1.set_ylabel(r'PV power(kW) & Temperature($^{\circ}$C)', fontsize=18)
        # ax1.plot(self.Ir_POA, '--', lw=2, color='mediumblue', label=' POA irradiance', zorder=3) 
        # # plt.suptitle("PV output according to POA Irradiance and Module Temperature ", fontsize = 14)
        # ax1.tick_params(which='major', axis='both', direction='in')
        # ax1.tick_params(which='minor', axis='both', direction='in')      
        # ax1.patch.set_visible(False)
        # ax1.set_zorder(1)

        # ax2 = ax1.twinx()   
        # ax2.bar(self.time_h, self.PV_stack, color='lightsteelblue', label='Total output', width=2)    
        # # ax2.plot(self.new_GHI, '-.', lw=2, label='new_GHI', zorder=3)
        # ax2.set_ylabel(r'Irradiance(W/${m^2}$) & Total output(kWh)', fontsize=18)
        # ax2.set_zorder(-1)

        # lines = []
        # labels = []
        # for ax in fig.axes:
        #     axLine , axLabel = ax.get_legend_handles_labels()
        #     lines.extend(axLine)
        #     labels.extend(axLabel)
        
        # ax1.legend(lines, labels, loc="center left", fontsize=18)
        



        plt.show()

PV_cal = Photovoltaic()
PV_cal.Local_Solar(Local)
PV_cal.POA(Angle)
PV_cal.Module_Tem(PV)
PV_cal.PV_Gen(PV)

result = np.stack((PV_cal.solar_altitude, PV_cal.solar_azimuth_2, PV_cal.solar_zenith, PV_cal.angle_of_incidence, PV_cal.new_GHI, PV_cal.DHI, PV_cal.DNI, PV_cal.POA_b, PV_cal.POA_d, PV_cal.POA_g, PV_cal.Ir_POA, PV_cal.new_Tem, PV_cal.T_m, PV_cal.EFF_norm, PV_cal.PV_output, PV_cal.real_PV, PV_cal.PV_stack), axis=1)
result = pd.DataFrame(result, columns = ['Altitude', 'Azimuth', 'Zenith', 'AOI', 'GHI', 'DHI', 'DNI', 'POA_b', 'POA_d', 'POA_g','Ir_POA', 'Tem', 'T_m', 'EFF_norm', 'PV_Gen', 'Real', 'Total'])

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(round(result, 4))
print(PV_cal.rmse)
print(PV_cal.MAE)

# error = np.zeros(60)
# for i in range(20, 80):
#     error[i-20] = PV_cal.relative_error[i]
# plt.figure()
# plt.plot(error, label='error')
# plt.legend()
# plt.tight_layout()
# plt.show()
