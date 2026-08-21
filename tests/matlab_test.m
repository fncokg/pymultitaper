rng(42);
fs = 8000;
duration_sec = 2;
n_samples = fs * duration_sec;
sig = randn(n_samples,1);
time_step = 0.05;
window_length = time_step;
nts = int16(time_step * fs);
nwl = nts;

frames = framesig(sig,nwl);

[sxx_eig,f_eig] = pmtm(frames,4,512,fs,"eigen");
[sxx_uni,f_uni] = pmtm(frames,4,512,fs,"unity");

sxx = vertcat(sxx_eig, sxx_uni);

fid = fopen("matlab_out.bin","w");
fwrite(fid,sxx,"double");
fclose(fid);

fid = fopen("matlab_in.bin","w");
fwrite(fid,sig,"double");
fclose(fid);