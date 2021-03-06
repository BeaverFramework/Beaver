Pod::Spec.new do |s|
  s.name         = "BeaverPromiseKit"
  s.version      = "0.1.0"
  s.summary      = "A delightful framework to build your iOS application"
  s.homepage     = "https://github.com/trupin/Beaver"
  s.license      = { :type => "MIT", :file => "LICENSE" }
  s.authors      = { "Theophane Rupin" => "theophane.rupin@gmail.com" }
  s.source       = { :git => "https://github.com/trupin/Beaver.git", :tag => "#{s.version}" }

  s.ios.deployment_target = "9.0"
  s.osx.deployment_target = "10.10"

  s.source_files = "BeaverPromiseKit/**/*.swift"

  s.dependency "PromiseKit", "~> 4.4"
  s.dependency "Beaver", "~> #{s.version}"
end
