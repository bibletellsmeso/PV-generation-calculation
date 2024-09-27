import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# https://bd.kma.go.kr/kma2020/fs/energySelect1.do?pageNum=5&menuCd=F050701000
Local = {'local_latitude' : 33.1446, 'local_longitude' : 126.3355, 'standard_longitude' : 135, 'year' : 2023, 'month' : 1, 'day' : 12, 'hour' : 12, 'min' : 0}
Angle = {'sun_k': 1355, 'elevation_angle': 30, 'panel_azimuth': 180, 'panel_tilted_angle': 30, 'zenith_angle': 60, 'rho': 0.31}
PV = {'alpha': -0.00415, 'gamma': 0.0325, 'EFF_STC': 1, 'Power_Capacity': 132.46}


# 각도 단위 변경
d2r = math.pi / 180
r2d = 180 / math.pi

class Photovoltaic:
    def __init__(self):
        self.Sim_time = 96 # 15min
        self.Weather_Forecast = np.genfromtxt('Weather Forecast.csv', delimiter=',', skip_header=3, dtype=np.float32)
        self.Ir = self.Weather_Forecast[:,3]
        self.Tem = self.Weather_Forecast[:,4]
        self.Predict_PV = self.Weather_Forecast[:,1]
        
        self.emtpy_Predict_PV = np.full(self.Sim_time, np.nan)
        self.empty_Ir = np.full(self.Sim_time, np.nan)
        self.empty_Tem = np.full(self.Sim_time, np.nan)
        
        for i in range(24):
            self.empty_Ir[4*i] = self.Ir[i]
            self.empty_Tem[4*i] = self.Tem[i]
            self.emtpy_Predict_PV[4*i] = self.Predict_PV[i]

        # 관측된 수평면 전일사량 (Global Horizontal Irradiance)
        self.new_Ir = np.array(pd.DataFrame(self.empty_Ir).interpolate()).ravel()
        self.new_Tem = np.array(pd.DataFrame(self.empty_Tem).interpolate()).ravel()
        self.Predict_PV = np.array(pd.DataFrame(self.emtpy_Predict_PV).interpolate()).ravel()

    def Local_Solar(self, Local):
        # 제주도 서귀포
        self.local_latitude = Local['local_latitude']               # 위도
        self.local_longitude = Local['local_longitude']             # 경도
        self.standard_longitude = Local['standard_longitude']       # 자오선
        self.year = Local['year']
        self.month = Local['month']
        self.day = Local['day']
        self.hour = list(range(self.Sim_time))
        self.hour = np.array(self.hour) * 0.25
        self.min = Local['min']

        # 균시차 (Equaiont of Time), 진태양시와 평균태양시의 차이
        self.day_of_year = datetime(self.year, self.month, self.day).timetuple().tm_yday
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

    def Tilted_Ir(self, Angle):
        self.panel_azimuth = Angle['panel_azimuth']                         # 방위각, 정남형(180) -45~45
        self.panel_tilted_angle = Angle['panel_tilted_angle']               # 패널경사각-에기연, 제주 24

        # 경사면 입사각(The Angle of Incidence) 방법 1
        self.angle_of_incidence = r2d * np.arccos(np.cos(d2r * self.solar_zenith) * np.cos(d2r * self.panel_tilted_angle) + np.sin(d2r * self.solar_zenith) * np.sin(d2r * self.panel_tilted_angle) * np.cos(d2r * self.solar_azimuth_2 - d2r * self.panel_azimuth))
        
        # # 방법 2
        # self.angle_of_incidence = np.arccos(np.sin(d2r * solar_altitude) * np.cos(d2r * self.panel_tilted_angle)\
        #     + np.cos(d2r * self.solar_azimuth_2 - d2r * self.panel_tilted_angle) * np.cos(d2r * solar_altitude)\
        #         * np.sin(d2r * self.panel_tilted_angle)) * r2d

        # 법선면 직달일사량 (Direct Normal Irradiance)
        self.Ir_dn = np.zeros(self.Sim_time)
        self.Io = Angle['sun_k']                                                       # 대기외 일사량, 태양 상수
        self.Ir_oH = self.Io * np.cos(d2r * self.solar_zenith)
        self.pp = self.new_Ir / self.Ir_oH

        for i in range(self.Sim_time):
            if 1540 * self.pp[i] - 470 > 860:
                self.Ir_dn[i] = 860
            elif 0 < 1540 * self.pp[i] - 470 <= 860:
                self.Ir_dn[i] = 1540 * self.pp[i] - 470
            else:
                self.Ir_dn[i] = 0

        # 관측값 없을 때
        # self.Ir_dn = 910 * np.sin(d2r * self.solar_altitude) + 0.25 * (910 * np.sin(2 * d2r * self.solar_altitude)) # 652.02

        # 경사면 직달일사량 (Plane of Array Beam component)
        self.Ir_sd = self.Ir_dn * np.cos(d2r * self.angle_of_incidence)

        # 경사면 확산일사량 (POA Sky-Diffuse component)
        self.Ir_s = self.new_Ir - self.Ir_dn * np.sin(d2r * self.solar_altitude)   # 수평면 확산일사량 (Diffuse Horizontal irradiance)
        self.Ir_ps = self.Ir_s * ((1 + np.cos(d2r * self.panel_tilted_angle))) / 2 + self.new_Ir * (0.12 * d2r * self.solar_zenith - 0.04) * (1 - np.cos(d2r * self.panel_tilted_angle)) / 2

        # self.Ir_ps = np.zeros(self.Sim_time)
        # for i in range(self.Sim_time):
        #     if self.new_Ir[i] >= self.Ir_sd[i] + np.sin(d2r * self.solar_altitude[i]):
        #         self.Ir_ps[i] = (self.new_Ir[i] - self.Ir_sd[i] - np.sin(d2r * self.solar_altitude[i])) * ((1 + np.cos(d2r * self.panel_tilted_angle))) / 2
        #     else:
        #         self.Ir_ps[i] = 0

        # 지표면 반사일사량 (Ground-Reflected component)
        self.rho = Angle['rho']
        self.Ir_gr = self.rho * self.new_Ir * ((1 - np.cos(d2r * self.panel_tilted_angle)) / 2)

        # 경사면 전일사량(Plane of Array Irradiance)
        self.Ir_t = self.Ir_sd + self.Ir_ps + self.Ir_gr

    def Module_Tem(self, PV):
        self.alpha = PV['alpha']   # \degc (crystalline Siicon (cSi)), -0.00415 (우리나라 모듈 출력온도계수 평균)
        self.gamma = PV['gamma']   # free-standing installations = 0.02 / roof integrated systems = 0.056
        self.T_m = self.new_Tem + self.gamma * self.Ir_t

    def Normal_EFF(self):
        for i in range(self.Sim_time):
            if self.new_Ir[i] <= 50:
                self.EFF_norm[i] = 0.91
            elif 50 < self.new_Ir[i] <= 150:
                self.EFF_norm[i] = (0.95-0.91) / (150-50) * (self.new_Ir[i] - 50) + 0.91
            elif 150 < self.new_Ir[i] <= 250:
                self.EFF_norm[i] = (0.98-0.95) / (250-150) * (self.new_Ir[i] - 150) + 0.95
            elif 250 < self.new_Ir[i] <= 350:
                self.EFF_norm[i] = (0.99-0.98) / (350-250) * (self.new_Ir[i] - 250) + 0.98
            elif 350 < self.new_Ir[i] <= 450:
                self.EFF_norm[i] = (1.00-0.99) / (450-350) * (self.new_Ir[i] - 350) + 0.99 
            elif 450 < self.new_Ir[i] <= 550:
                self.EFF_norm[i] = (1.02-1.00) / (550-450) * (self.new_Ir[i] - 450) + 1.00
            elif 550 < self.new_Ir[i] <= 650:
                self.EFF_norm[i] = (1.03-1.02) / (650-550) * (self.new_Ir[i] - 550) + 1.02
            elif 650 < self.new_Ir[i] <= 750:
                self.EFF_norm[i] = (1.04-1.03) / (750-650) * (self.new_Ir[i] - 650) + 1.03
            elif 750 < self.new_Ir[i] <= 850:
                self.EFF_norm[i] = (1.03-1.04) / (850-750) * (self.new_Ir[i] - 750) + 1.04
            elif 850 < self.new_Ir[i] <= 950:
                self.EFF_norm[i] = (1.01-1.03) / (950-850) * (self.new_Ir[i] - 850) + 1.03
            elif 950 < self.new_Ir[i] <= 1050:
                self.EFF_norm[i] = (0.99-1.01) / (1050-950) * (self.new_Ir[i] - 950) + 1.01
            elif 1050 < self.new_Ir[i] <= 1150:
                self.EFF_norm[i] = (0.98-0.99) / (1150-1050) * (self.new_Ir[i] - 1050) + 0.99
            elif 1150 < self.new_Ir[i] <= 1250:
                self.EFF_norm[i] = (0.97-0.98) / (1250-1150) * (self.new_Ir[i] - 1150) + 0.98
            else:
                self.EFF_norm[i] = 0.97
        return self.EFF_norm

    def PV_Gen(self, PV):
        self.alpha = PV['alpha']
        self.P_nom =  PV['Power_Capacity']

        # self.a_1, self.a_2, self.a_3 = 1, 2, 3
        # self.EFF_25 = self.EFF_25 = self.a_1 + self.a_2 * self.new_Ir + self.a_3 * math.log10(self.new_Ir)
        # self.EFF_t = self.EFF_25 * (1 + self.alpha * (self.T_m - 25))
        # self.EFF_STC = PV['EFF_STC']        # 18 % (Polycrystalline silicon module)

        self.EFF_norm = np.zeros(self.Sim_time)
        self.EFF_norm = self.Normal_EFF()
        self.PV_output = self.EFF_norm * (1 + self.alpha * (self.T_m - 25)) * (self.Ir_t / 1000) * self.P_nom   
        self.PV_output_g = self.EFF_norm * (1 + self.alpha * (self.new_Tem - 25)) * (self.new_Ir / 1000) * self.P_nom

        self.PV_stack = np.zeros(96)
        for i in range(24-1):
                self.PV_stack[4*(i+1)] = self.PV_stack[4*i] + self.PV_output[4*i] + self.PV_output[4*i+1] + self.PV_output[4*i+2] + self.PV_output[4*i+3]
        self.time_h = np.zeros(self.Sim_time)
        for i in range(self.Sim_time-1):
            self.time_h[0] = 0
            self.time_h[i+1] = self.time_h[i] + 1

        # plt.plot(self.PV_capacity, 'b-', self.PV_output, 'ro-,', self.T_m)

        plt.style.use(['science', 'no-latex'])

        # 고도 및 방위각
        fig = plt.figure(1)
        plt.grid(color='lightsteelblue', which='major', axis='both', linestyle='--', linewidth=1)
        plt.grid(color='lightsteelblue', which='minor', axis='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Time step(15m)')
        plt.ylabel('Degree')
        plt.suptitle("Solar Altitude and Azimuth", fontsize = 14)
        plt.plot(self.solar_altitude, '-', lw=2, color='mediumblue', label = 'Altitude')
        plt.plot(self.solar_azimuth_2, '-', lw=2, color='crimson', label = 'Azimuth')
        fig.legend(loc='upper right')

        # 일사량 비교
        fig = plt.figure(2, figsize=(12,9))
        plt.grid(color='lightsteelblue', which='major', axis='both', linestyle='--', linewidth=1)
        plt.grid(color='lightsteelblue', which='minor', axis='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Time step(15m)', fontsize=12)
        plt.ylabel(r'Irradiance(W/${m^2}$)', fontsize=12)
        plt.suptitle("Horizontal and Plane of array Irradiance", fontsize = 14)
        plt.plot(self.new_Ir, '--', lw=2, marker=10, color='mediumblue', label = 'Ir_global', zorder=2)
        plt.plot(self.Ir_t, '-', lw=2, marker=11, color='crimson', label = 'Ir_local', zorder=1)
        fig.legend()
        # plt.rcParams['font.family'] = 'Malgun Gothic'
        # plt.rcParams['axes.unicode_minus'] = False

        # 온도 비교
        fig = plt.figure(3, figsize=(12,9))
        plt.grid(color='lightsteelblue', which='major', axis='both', linestyle='--', linewidth=1)
        plt.grid(color='lightsteelblue', which='minor', axis='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Time step(15m)', fontsize=12)
        plt.ylabel(r'Temperature($^{\circ}$C)', fontsize=12)
        plt.suptitle("Temperature and Module Temperature", fontsize = 14)
        plt.plot(self.new_Tem, '--', lw=2, color='mediumblue', label = 'T_global', zorder=2)
        plt.plot(self.T_m, '-', lw=2, color='crimson', label = 'T_local', zorder=1)
        fig.legend()

        # 일사량 및 온도와 발전량 관계
        fig, ax1 = plt.subplots(1, figsize=(12,9))    # 직접 figure 객체를 생성
        ax1.plot(self.PV_output / np.max(self.PV_output), 'o-', lw=2, color='crimson', label='PV output', zorder=2)
        ax1.plot(self.Ir_t / np.max(self.Ir_t), '-', lw=2, color='mediumblue', label='POA irradiance', zorder=3)
        ax1.plot(self.T_m / np.max(self.T_m), '-', lw=2, color='#FA7268', label='Module temperate', zorder=1)
        ax1.set_xlabel('Time step(15m)', fontsize=12)
        ax1.set_ylabel('Normalized irradiance & PV power', fontsize=12)
        plt.suptitle("PV Output Correlation", fontsize = 14)
        ax1.minorticks_on()
        ax1.grid(color='lightsteelblue',  which='major', linestyle = '--', linewidth = 0.5)
        ax1.grid(color='lightsteelblue',  which='minor', linestyle = '--', linewidth = 0.5)
        ax1.tick_params(which='major', axis='both', direction='in')
        ax1.tick_params(which='minor', axis='both', direction='in')      
        ax1.patch.set_visible(False)
        ax1.set_zorder(1)
        ax1.legend(loc="best")
        
        # 발전량 계산
        fig, ax1 = plt.subplots(1, figsize=(12,9))
        ax1.plot(self.PV_output, '-', marker='o', lw=2, color='crimson', label='PV output_local', zorder=2)
        # ax1.plot(self.Predict_PV, '--', lw=2, color='forestgreen', alpha=0.7, label='Predcition', zorder=1)
        # ax1.plot(self.PV_output_g, '-', lw=2, marker=10, color='mediumblue', label='PV output_global', zorder=0)
        ax1.plot(self.T_m, '-.', lw=2, color='#FA7268', label='Module temperate', zorder=1)
        ax1.hlines(y=self.P_nom, xmin=0, xmax=96, color='blueviolet', linestyles='solid', label='Capacity', lw=2, zorder=1)
        ax1.set_xlabel('Time step(15m)', fontsize=12)
        ax1.set_ylabel(r'PV power(kW) & Temperature($^{\circ}$C)', fontsize=12)
        plt.suptitle("PV output according to POA Irradiance and Module Temperature ", fontsize = 14)
        ax1.minorticks_on()
        ax1.grid(color='lightsteelblue',  which='major', linestyle = '--', linewidth = 1)
        # ax1.grid(color='lightsteelblue',  which='minor', linestyle = '--', linewidth = 0.5)
        ax1.tick_params(which='major', axis='both', direction='in')
        ax1.tick_params(which='minor', axis='both', direction='in')      
        ax1.patch.set_visible(False)
        ax1.set_zorder(1)

        ax2 = ax1.twinx()
        ax2.bar(self.time_h, self.PV_stack, color='lightsteelblue', label='Total output', width=2)
        ax2.plot(self.Ir_t, '--', lw=2, color='mediumblue', label=' POA irradiance', zorder=3)     
        # ax2.plot(self.new_Ir, '-.', lw=2, label='GHI', zorder=3)
        ax2.set_ylabel(r'Irradiance(W/${m^2}$) & Total output(kWh)', fontsize=12)
        ax2.set_zorder(-1)

        lines = []
        labels = []
        for ax in fig.axes:
            axLine , axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)
        
        ax1.legend(lines, labels, loc="center left")

        plt.show()

PV_cal = Photovoltaic()
PV_cal.Local_Solar(Local)
PV_cal.Tilted_Ir(Angle)
PV_cal.Module_Tem(PV)
PV_cal.PV_Gen(PV)

result = np.stack((PV_cal.solar_altitude, PV_cal.solar_azimuth, PV_cal.solar_zenith, PV_cal.new_Ir, PV_cal.Ir_sd, PV_cal.Ir_ps, PV_cal.Ir_gr, PV_cal.Ir_t, PV_cal.new_Tem, PV_cal.T_m, PV_cal.EFF_norm, PV_cal.PV_output, PV_cal.PV_stack), axis=1)
result = pd.DataFrame(result, columns = ['Altitude', 'Azimuth', 'Zenith', 'Ir', 'Ir_sd', 'Ir_ps', 'Ir_gr', 'Ir_t', 'Tem', 'T_m', 'EFF_norm', 'PV_Gen', 'Total'])

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(round(result, 4))
