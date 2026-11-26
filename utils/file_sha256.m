function hash=file_sha256(path)
% Binary SHA-256 for reproducibility manifests; read-only source access.
fid=fopen(path,'rb');
if fid<0, error('STAR_KF:Fingerprint','Cannot read %s.',path); end
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
md=java.security.MessageDigest.getInstance('SHA-256');
while ~feof(fid)
    bytes=fread(fid,1048576,'*uint8');
    if ~isempty(bytes), md.update(bytes); end
end
hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end
