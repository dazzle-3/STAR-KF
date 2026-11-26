function log_progress(tag, message, varargin)
if nargin < 2 || isempty(message)
    return;
end

if nargin < 1 || isempty(tag)
    tag = 'INFO';
end

if nargin > 2
    text = sprintf(message, varargin{:});
else
    text = message;
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf('[%s] [%s] %s\n', timestamp, upper(char(string(tag))), text);
end
